#!/usr/bin/env python3
"""
process_audio.py

Prototype: process an audio file, run a VAD / noise classifier per short frame,
and output JSON with labeled timestamp ranges.

Usage:
  python process_audio.py \
    --audio examples/test.wav \
    --model_path path/to/humaware_vad.jit \
    --out timestamps.json \
    --frame_sec 1.0 \
    --hop_sec 0.5

Notes:
- The script tries to load `model_path` as a TorchScript model first.
- If `model_path` is omitted or not a torchscript, and transformers is installed,
  the script will attempt to use Hugging Face `pipeline("audio-classification", ...)`.
- The model output format varies across models; this script assumes the model
  returns per-frame logits/probabilities or a label per input frame. You may
  need to adapt `run_model_on_frame()` for your specific model's output shape.
"""

import argparse
import json
import math
import os
from typing import List, Dict, Tuple, Optional
from config import *
import numpy as np
import soundfile as sf

# Optional imports (import errors handled gracefully)
try:
    import torch
except Exception:
    torch = None

try:
    import torchaudio
except Exception:
    torchaudio = None

try:
    from transformers import pipeline
except Exception:
    pipeline = None


# -----------------------------
# Math / framing explanation
# -----------------------------
#
# Let fs be the sample rate (samples per second).
# Let frame_sec be frame length in seconds.
# Let hop_sec be hop (stride) in seconds.
#
# Then:
#   frame_samples = round(frame_sec * fs)         (number of samples per frame)
#   hop_samples   = round(hop_sec * fs)           (samples between successive frame starts)
#
# Frame i (0-based) starts at sample:
#   start_sample_i = i * hop_samples
# and ends at sample:
#   end_sample_i = start_sample_i + frame_samples - 1
#
# Convert sample index to time (seconds):
#   t_seconds = sample_index / fs
#
# So frame i start time in seconds:
#   start_time_i = (i * hop_samples) / fs
# and frame i end time:
#   end_time_i = ((i * hop_samples) + frame_samples) / fs
#
# If model returns a detection within a frame with relative offset r_start..r_end (seconds),
# the absolute timestamp is:
#   abs_start = start_time_i + r_start
#   abs_end   = start_time_i + r_end
#
# When merging adjacent frames into contiguous ranges, we merge if gap <= merge_tolerance_sec.
#
# -----------------------------


# -----------------------------
# Helpers: audio loading and framing
# -----------------------------
def load_audio_mono(path: str, target_fs: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """
    Load audio file (any channels) and return mono float32 numpy array and sample rate.
    Uses soundfile for robust file support.

    Returns:
      waveform: shape (n_samples,), dtype float32, in range [-1.0, 1.0]
      sr: sample rate (int)
    """
    data, sr = sf.read(path, dtype="float32")  # data shape: (n_samples,) or (n_samples, channels)
    if data.ndim == 2:
        # downmix to mono by averaging channels
        data = data.mean(axis=1)
    if target_fs is not None and target_fs != sr:
        if torchaudio is None or torch is None:
            raise RuntimeError("Resampling requested but torchaudio/torch not available.")
        # resample via torchaudio (tensor-based)
        tensor = torch.from_numpy(data).unsqueeze(0)  # shape 1 x n
        resampler = torchaudio.transforms.Resample(sr, target_fs)
        with torch.no_grad():
            tensor_rs = resampler(tensor)
        data = tensor_rs.squeeze(0).numpy()
        sr = target_fs
    return data.astype(np.float32), sr


def frame_generator(waveform: np.ndarray, sr: int, frame_sec: float, hop_sec: float):
    """
    Yield frames with metadata.
    Yields tuples:
      (frame_index, frame_np: np.ndarray, start_sample, end_sample, start_time, end_time)
    """
    frame_sample_length = int(round(frame_sec * sr))
    hop_sample_length = int(round(hop_sec * sr))
    total_sample_length = waveform.shape[0]
    
    # number of frames (include last partial frame if it has at least 1 sample)
    if total_sample_length <= 0:
        return
    n_frames = max(1, 1 + (total_sample_length - frame_sample_length + hop_sample_length - 1) // hop_sample_length) \
        if total_sample_length > frame_sample_length else 1

    for i in range(n_frames):
        start = i * hop_sample_length
        end = start + frame_sample_length  # exclusive
        # Clip to array bounds
        if start >= total_sample_length:
            break
        frame = waveform[start:min(end, total_sample_length)]
        # If last frame is shorter than frame_samples, optionally pad with zeros for model input
        if frame.shape[0] < frame_sample_length:
            pad_width = frame_sample_length - frame.shape[0]
            frame = np.pad(frame, (0, pad_width), mode="constant", constant_values=0.0)
        start_time = start / sr
        end_time = min(end, total_sample_length) / sr
        yield i, frame, start, min(end, total_sample_length), start_time, end_time


# -----------------------------
# Model runner abstraction
# -----------------------------
class ModelRunner:
    """
    Abstract model runner that provides `infer_frame(frame: np.ndarray, sr: int) -> List[Dict]`.
    Each dict in returned list is expected to be:
      { "label": str, "start": float, "end": float, "score": float }
    where start/end are relative to frame (seconds).
    """

    def __init__(self, model_path: Optional[str] = None, model_sample_rate: int = CONFIG_AUDIO_SR):
        self.model_path = model_path
        self.sr = model_sample_rate
        self.torchscript = None
        self.hf_pipe = None

        # Try loading torchscript if possible
        if model_path and torch is not None:
            try:
                self.torchscript = torch.jit.load(model_path, map_location="cpu")
                self.torchscript.eval()
                print(f"[ModelRunner] Loaded TorchScript model from {model_path}")
            except Exception as e:
                print(f"[ModelRunner] Could not load TorchScript model ({e}), will try HF pipeline if available.")
                self.torchscript = None

        # Optionally configure HF pipeline if transformers available and model_path looks like a repo id
        if self.torchscript is None and pipeline is not None and model_path:
            try:
                # This attempts to create an audio-classification pipeline. It may or may not match the repo.
                self.hf_pipe = pipeline("audio-classification", model=model_path, chunk_length_s=1.0, device=-1)
                print(f"[ModelRunner] Created HF audio-classification pipeline for {model_path}")
            except Exception as e:
                print(f"[ModelRunner] Could not create HF pipeline ({e}). No model available.")
                self.hf_pipe = None

        if self.torchscript is None and self.hf_pipe is None:
            print("[ModelRunner] No model loaded. Running in 'dummy' mode (random/no-op).")

    def infer_frame(self, frame: np.ndarray, sr: int) -> List[Dict]:
        """
        Infer on a single frame. Frame is a 1D float32 numpy array, length == model expected samples.

        Returns a list of detections:
          [{ "label": "hum"|"speech"|"silence", "start": 0.0, "end": 1.0, "score": 0.9 }, ...]

        NOTE: You will almost certainly need to adapt this function to match your model's I/O.
        """
        # If TorchScript model loaded, attempt a simple call.
        if self.torchscript is not None and torch is not None:
            # Convert numpy frame to tensor expected by model:
            # Many audio models expect shape [1, n_samples] float32
            wav = torch.from_numpy(frame).unsqueeze(0)  # 1 x n
            with torch.no_grad():
                out = self.torchscript(wav, sr)  # model-specific output
            # Heuristic post-processing:
            # If out is a dict-like with 'probs' or 'logits', adapt accordingly.
            # Here we try to handle some common shapes. IMPORTANT: adapt to your model.
            if isinstance(out, dict):
                # e.g., {"logits": tensor([..., ...])}
                if "logits" in out:
                    probs = torch.softmax(out["logits"], dim=-1).cpu().numpy()
                    # pick argmax label
                    class_idx = int(probs.argmax(axis=-1).item())
                    score = float(probs.max())
                    label = f"class_{class_idx}"
                    return [{"label": label, "start": 0.0, "end": frame.shape[0] / sr, "score": score}]
            elif isinstance(out, torch.Tensor):
                # single-tensor outputs: interpret as per-frame scores or class logits
                arr = out.cpu().numpy()
                # If 1D single value -> treat above threshold as "detected"
                if arr.size == 1:
                    score = float(arr.item())
                    label = "detected" if score > 0.5 else "not"
                    return [{"label": label, "start": 0.0, "end": frame.shape[0] / sr, "score": score}]
                # If vector of class logits:
                if arr.ndim >= 1:
                    probs = softmax(arr.flatten())
                    idx = int(np.argmax(probs))
                    score = float(np.max(probs))
                    label = f"class_{idx}"
                    return [{"label": label, "start": 0.0, "end": frame.shape[0] / sr, "score": score}]
            # Unknown output; fallback:
            return []
        # If HF pipeline present, call it (it returns list of dicts with label/score)
        if self.hf_pipe is not None:
            # HF pipeline can accept numpy arrays directly as {"array": np.array, "sampling_rate": sr}
            try:
                # The `chunk_length_s` param controls pipeline chunking; we provide the frame directly
                results = self.hf_pipe(frame, sampling_rate=sr)
                # Example HF result: [{"label":"SPEECH","score":0.98}, ...]
                detections = []
                for r in results:
                    # Choose label and score, whole-frame span
                    detections.append({
                        "label": r.get("label", "unknown").lower(),
                        "start": 0.0,
                        "end": frame.shape[0] / sr,
                        "score": float(r.get("score", 0.0))
                    })
                return detections
            except Exception as e:
                print(f"[ModelRunner] HF pipeline inference failed: {e}")
                return []
        # Dummy fallback: return empty (no labels) or simple energy-based silence detection
        energy = float(np.mean(np.square(frame)))
        # simple thresholds (tunable)
        if energy < 1e-6:
            return [{"label": "silence", "start": 0.0, "end": frame.shape[0] / sr, "score": 1.0}]
        elif energy < 1e-3:
            return [{"label": "hum", "start": 0.0, "end": frame.shape[0] / sr, "score": min(0.99, (1e-3 - energy) / 1e-3)}]
        else:
            return [{"label": "speech", "start": 0.0, "end": frame.shape[0] / sr, "score": min(0.99, energy * 1000)}]


def softmax(x: np.ndarray):
    e = np.exp(x - np.max(x))
    return e / e.sum(axis=-1, keepdims=True)


# -----------------------------
# Post-processing: merge frame detections into continuous timestamp ranges
# -----------------------------
def map_frame_detections_to_absolute(
    detections_per_frame: List[Tuple[int, List[Dict], float]],
    frame_sec: float,
    hop_sec: float,
    sr: int,
    merge_tolerance_sec: float = 0.15
) -> List[Dict]:
    """
    Convert list of (frame_idx, detections, frame_start_time) into merged absolute ranges.

    detections_per_frame: list of tuples (i, detections, frame_start_time_seconds)
    detections: [ {"label":str, "start":rel_start, "end":rel_end, "score":float}, ... ]
    """
    absolute_segments = []
    for i, dets, frame_start in detections_per_frame:
        for d in dets:
            abs_start = frame_start + float(d.get("start", 0.0))
            abs_end = frame_start + float(d.get("end", frame_sec))
            segment = {
                "label": d.get("label", "unknown"),
                "start": float(abs_start),
                "end": float(abs_end),
                "score": float(d.get("score", 0.0))
            }
            absolute_segments.append(segment)
    if not absolute_segments:
        return []

    # Sort by start
    absolute_segments.sort(key=lambda x: x["start"])

    # Merge adjacent segments with same label when close or overlapping
    merged = [absolute_segments[0].copy()]
    for seg in absolute_segments[1:]:
        last = merged[-1]
        # If same label and gap <= merge_tolerance, merge
        if seg["label"] == last["label"] and seg["start"] <= last["end"] + merge_tolerance_sec:
            # extend end and pick max score
            last["end"] = max(last["end"], seg["end"])
            last["score"] = max(last["score"], seg["score"])
        else:
            merged.append(seg.copy())
    return merged


# -----------------------------
# CLI main
# -----------------------------
def main():
    p = argparse.ArgumentParser(description="Process audio file and output labeled timestamps as JSON.")
    p.add_argument("--audio", required=True, help="Input audio file (wav, flac, mp3, ...)")
    p.add_argument("--model_path", default=None, help="Path to TorchScript model or HF model repo id")
    p.add_argument("--out", default="timestamps.json", help="Output JSON file")
    p.add_argument("--frame_sec", type=float, default=1.0, help="Frame length in seconds")
    p.add_argument("--hop_sec", type=float, default=0.5, help="Hop (stride) in seconds")
    p.add_argument("--model_sr", type=int, default=16000, help="Target sample rate for model")
    p.add_argument("--merge_tol", type=float, default=0.15, help="Merge tolerance in seconds")
    args = p.parse_args()
    results = execute(args)
    # 6) Write JSON
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[segment_audio] Wrote output to {args.out}")

def execute(args):
    # 1) Load audio and resample to model sr if needed
    print(f"[segment_audio] Loading audio: {args.audio}")
    waveform, sr = load_audio_mono(args.audio, target_fs=args.model_sr)
    n_samples = waveform.shape[0]
    duration = n_samples / sr
    print(f"[segment_audio] audio samples={n_samples}, sr={sr}, duration={duration:.3f}s")

    # 2) Initialize model runner
    runner = ModelRunner(model_path=args.model_path, model_sample_rate=args.model_sr)

    # 3) Iterate frames, run model, collect per-frame detections
    detections_per_frame = []
    for i, frame, s_sample, e_sample, s_time, e_time in frame_generator(waveform, sr, args.frame_sec, args.hop_sec):
        # call model
        dets = runner.infer_frame(frame, sr)
        # save frame start time for mapping
        detections_per_frame.append((i, dets, s_time))
        print(f"[frame {i}] start={s_time:.3f}s end={e_time:.3f}s detections={len(dets)}")

    # 4) Map to absolute segments and merge
    merged = map_frame_detections_to_absolute(detections_per_frame, args.frame_sec, args.hop_sec, sr, merge_tolerance_sec=args.merge_tol)
    print(f"[segment_audio] merged segments: {len(merged)}")

    # 5) Optionally filter segments by label / score threshold (example: only hum/silence)
    # For prototype we keep everything.
    results = {
        "audio_file": os.path.basename(args.audio),
        "duration": duration,
        "sample_rate": sr,
        "segments": merged
    }
    return results


if __name__ == "__main__":
    main()
