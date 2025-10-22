#!/usr/bin/env python3
"""
FastAPI server for HumSkip Tampermonkey integration.

Endpoints:
  POST /infer_chunk
    - form fields:
        - audio_blob: uploaded file (webm/opus, wav, or raw PCM blob)
        - video_time: float (start time of the chunk in seconds)
        - sample_rate: optional int (if the blob is raw PCM and you want to tell server)
    - returns JSON:
      {
        "labels": [
          {"label": "hum", "start": 0.0, "end": 0.9, "score": 0.92},
          ...
        ],
        "frame_start": 12.345   # the video_time echoed back
      }

Notes:
- This server decodes the uploaded audio to mono PCM16 WAV @ model_sr (default 16000).
- It then calls `process_audio_bytes(pcm16_bytes, model_sr, frame_start)` which
  should be provided by your own processing module (adapt the import accordingly).
- The server maps relative segment times returned by process_audio_bytes to absolute
  video timestamps before sending them back to the client.
"""

import os
import shutil
import subprocess
import tempfile
from typing import List

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- CONFIG ---
MODEL_SAMPLE_RATE = 16000  # server-side standard sampling rate for models
FFMPEG_BIN = shutil.which("ffmpeg") or "ffmpeg"  # ensure ffmpeg is in PATH
# --------------

app = FastAPI(title="HumSkip Local Inference (FastAPI)")

# Allow all origins for development; restrict in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LabelOut(BaseModel):
    label: str
    start: float  # relative to chunk (seconds) OR absolute? we'll return absolute below
    end: float
    score: float = 1.0


class InferResponse(BaseModel):
    labels: List[LabelOut]
    frame_start: float


# --------- User-provided processing wrapper (adapt as needed) ----------
# Import or wrap your actual processing function here. Example stub below.
# Replace the stub with: from my_processing_module import process_audio_bytes
def process_audio_bytes(pcm16_bytes: bytes, sample_rate: int, chunk_start_time: float):
    """
    USER-SUPPLIED: Replace this with your implementation.
    Expected to return a list of dicts: {"label":str, "start":float, "end":float, "score":float}
    with start/end **relative** to the chunk (seconds).
    """
    # --- Example dummy behavior: silence/hum/speech by energy (replace me) ---
    import numpy as np
    arr = np.frombuffer(pcm16_bytes, dtype="<i2").astype("float32") / 32768.0
    energy = float((arr ** 2).mean())
    dur = len(arr) / sample_rate
    if energy < 1e-6:
        return [{"label": "silence", "start": 0.0, "end": dur, "score": 1.0}]
    elif energy < 1e-3:
        return [{"label": "hum", "start": 0.0, "end": dur, "score": min(0.99, (1e-3 - energy) / 1e-3)}]
    else:
        return [{"label": "speech", "start": 0.0, "end": dur, "score": min(0.99, energy * 1000)}]
# ----------------------------------------------------------------------


def _ffmpeg_decode_to_pcm16_wav(input_path: str, output_path: str, target_sr: int = MODEL_SAMPLE_RATE):
    """
    Use ffmpeg to decode `input_path` (any container, webm/opus, mp3, wav, raw) into:
      - WAV, mono, PCM16, sample rate = target_sr

    This raises subprocess.CalledProcessError on failure.
    """
    cmd = [
        FFMPEG_BIN,
        "-y",                  # overwrite
        "-i", input_path,      # input file
        "-vn",                 # no video
        "-ac", "1",            # mono
        "-ar", str(target_sr), # resample
        "-sample_fmt", "s16",  # PCM16
        "-f", "wav",
        output_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


@app.post("/infer_chunk", response_model=InferResponse)
async def infer_chunk(
    audio_blob: UploadFile = File(...),
    video_time: float = Form(...),
    sample_rate: int = Form(None),
):
    """
    Accept an audio chunk and a video_time (chunk start time, seconds).
    Returns list of labeled segments (absolute timestamps).
    """
    # 1) Save uploaded file to a temp file
    tmp_dir = tempfile.mkdtemp(prefix="humskip_")
    try:
        uploaded_path = os.path.join(tmp_dir, "upload")
        with open(uploaded_path, "wb") as f:
            f.write(await audio_blob.read())

        # 2) Decide whether uploaded data is raw PCM or needs decoding
        # Heuristics: if client sent sample_rate param and filename indicates raw blob (no typical container)
        # For robustness we attempt ffmpeg decode unconditionally (ffmpeg handles raw PCM poorly unless told)
        decoded_wav = os.path.join(tmp_dir, "decoded.wav")
        try:
            _ffmpeg_decode_to_pcm16_wav(uploaded_path, decoded_wav, target_sr=MODEL_SAMPLE_RATE)
        except subprocess.CalledProcessError as e:
            # If decode failed, maybe it's already raw PCM16 bytes sent directly.
            # If sample_rate provided, trust it and use the uploaded bytes as PCM16 LE.
            if sample_rate:
                # treat uploaded_path as raw pcm16 little-endian samples
                with open(uploaded_path, "rb") as rf:
                    pcm16_bytes = rf.read()
                # call user function directly with these bytes
                rel_segments = process_audio_bytes(pcm16_bytes, sample_rate, float(video_time))
                # Map and return below
            else:
                raise HTTPException(status_code=400, detail="Could not decode uploaded audio; ensure ffmpeg supports the format or provide sample_rate for raw PCM.")
        else:
            # 3) If decoding succeeded, read PCM16 samples from decoded_wav
            with open(decoded_wav, "rb") as wf:
                # We want raw PCM16 bytes without WAV header. Use ffmpeg to output raw PCM would be another option,
                # but simplest here: use Python's wave module to extract frames, or call ffmpeg to output raw s16le bytes.
                import wave
                with wave.open(decoded_wav, "rb") as wavesrc:
                    # Ensure format expectations
                    channels = wavesrc.getnchannels()
                    sampwidth = wavesrc.getsampwidth()
                    sr = wavesrc.getframerate()
                    nframes = wavesrc.getnframes()
                    frames = wavesrc.readframes(nframes)
                    # If stereo, we decoded to mono via ffmpeg -ac 1 so channels==1
                    # frames is PCM16 LE if sampwidth==2
                    if sampwidth != 2:
                        raise HTTPException(status_code=500, detail="Unexpected sample width in decoded WAV.")
                    pcm16_bytes = frames
                    # pass to user function with model SR
                    rel_segments = process_audio_bytes(pcm16_bytes, sr, float(video_time))

        # 4) Map relative segments (returned by process_audio_bytes) to absolute timestamps
        # Formula (explicit):
        #   abs_start = video_time + rel_start
        #   abs_end   = video_time + rel_end
        abs_segments = []
        for seg in rel_segments:
            label = seg.get("label", "unknown")
            rel_start = float(seg.get("start", 0.0))
            rel_end = float(seg.get("end", 0.0))
            score = float(seg.get("score", 0.0))
            abs_segments.append({
                "label": label,
                # absolute times in seconds relative to video
                "start": video_time + rel_start,
                "end": video_time + rel_end,
                "score": score,
            })

        # 5) Optionally: merge adjacent same-label segments server-side (simple merge)
        # For simplicity we return what the model gave (client also merges), but you can merge here.
        return {"labels": abs_segments, "frame_start": video_time}

    finally:
        # cleanup
        try:
            shutil.rmtree(tmp_dir)
        except Exception:
            pass


@app.post("/preprocess")
async def preprocess_file(audio_file: UploadFile = File(...)):
    """
    Optional endpoint to upload a full audio file for offline processing.
    This demonstrates how you might pipeline a full-file batch run using your
    existing processing functions.
    Returns: small manifest / OK.
    """
    # save file
    tmp_dir = tempfile.mkdtemp(prefix="humskip_pre_")
    try:
        path = os.path.join(tmp_dir, "full_input")
        with open(path, "wb") as f:
            f.write(await audio_file.read())

        # decode to model SR with ffmpeg
        decoded = os.path.join(tmp_dir, "decoded.wav")
        _ffmpeg_decode_to_pcm16_wav(path, decoded, target_sr=MODEL_SAMPLE_RATE)

        # Here you could call your existing batch processing: e.g. process_whole_file(decoded)
        # For now we simply acknowledge receipt.
        return {"status": "ok", "note": "file received and decoded to model sample rate. Implement batch processing as needed."}
    finally:
        try:
            shutil.rmtree(tmp_dir)
        except Exception:
            pass