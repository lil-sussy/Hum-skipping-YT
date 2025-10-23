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
import uvicorn
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

# --- paste into your main.py (replace previous infer_chunk) ---
from fastapi import Request

@app.post("/infer_chunk")
async def infer_chunk(request: Request):
    """
    Robust /infer_chunk endpoint supporting:
      - multipart/form-data with fields:
          audio_blob (file), video_time (form), sample_rate (optional form)
      - raw POST body (application/octet-stream) with query param:
          /infer_chunk?video_time=12.34[&sample_rate=16000]
        -> body is treated as raw PCM16 bytes if sample_rate provided,
           otherwise body is treated as an encoded container and will be
           written to a temp file and ffmpeg-decoded.
    Returns JSON:
      {"labels":[{"label":...,"start":abs_seconds,"end":abs_seconds,"score":...}, ...],
       "frame_start": video_time}
    """
    import io, wave, json
    tmp_dir = None
    try:
        content_type = (request.headers.get("content-type") or "").lower()
        # Case A: multipart/form-data (typical browser FormData)
        if "multipart/form-data" in content_type:
            form = await request.form()
            # form can contain UploadFile-like object under 'audio_blob'
            if "video_time" not in form:
                raise HTTPException(status_code=400, detail="multipart missing 'video_time' field")
            video_time = float(form.get("video_time"))
            sample_rate = form.get("sample_rate")
            upload = form.get("audio_blob")
            if upload is None:
                raise HTTPException(status_code=400, detail="multipart missing 'audio_blob' file")
            # upload may be starlette UploadFile or a plain bytes-like object
            try:
                # UploadFile has .file or .read
                content = await upload.read()
            except Exception:
                # fallback
                content = upload.file.read() if hasattr(upload, "file") else bytes(upload)
            # We'll write content to temp file and decode via ffmpeg (most reliable)
            tmp_dir = tempfile.mkdtemp(prefix="humskip_")
            uploaded_path = os.path.join(tmp_dir, "upload")
            with open(uploaded_path, "wb") as f:
                f.write(content)
            # Try decode with ffmpeg -> decoded wav path
            decoded_wav = os.path.join(tmp_dir, "decoded.wav")
            try:
                _ffmpeg_decode_to_pcm16_wav(uploaded_path, decoded_wav, target_sr=MODEL_SAMPLE_RATE)
                # read PCM from decoded wav
                with wave.open(decoded_wav, "rb") as wavesrc:
                    frames = wavesrc.readframes(wavesrc.getnframes())
                    pcm16_bytes = frames
                    sr = wavesrc.getframerate()
            except subprocess.CalledProcessError:
                # if decode fails, but sample_rate was provided, assume raw PCM16
                if sample_rate:
                    with open(uploaded_path, "rb") as rf:
                        pcm16_bytes = rf.read()
                    sr = int(sample_rate)
                else:
                    raise HTTPException(status_code=400, detail="ffmpeg failed to decode uploaded file; provide raw PCM with sample_rate or upload a supported container.")
        else:
            # Case B: raw POST body (easier for Tampermonkey)
            # Expect video_time in query string
            q = dict(request.query_params)
            if "video_time" not in q:
                raise HTTPException(status_code=400, detail="Missing query parameter 'video_time' for raw body POST. Example: /infer_chunk?video_time=12.34")
            video_time = float(q["video_time"])
            sample_rate = q.get("sample_rate")
            body = await request.body()
            # If sample_rate provided: treat body as raw PCM16 little-endian
            if sample_rate:
                pcm16_bytes = body
                sr = int(sample_rate)
            else:
                # Otherwise treat body as encoded container; write to temp file and decode
                tmp_dir = tempfile.mkdtemp(prefix="humskip_")
                uploaded_path = os.path.join(tmp_dir, "upload")
                with open(uploaded_path, "wb") as f:
                    f.write(body)
                decoded_wav = os.path.join(tmp_dir, "decoded.wav")
                try:
                    _ffmpeg_decode_to_pcm16_wav(uploaded_path, decoded_wav, target_sr=MODEL_SAMPLE_RATE)
                except subprocess.CalledProcessError:
                    raise HTTPException(status_code=400, detail="ffmpeg failed to decode raw POST body; provide sample_rate if sending raw PCM, or POST a supported container.")
                import wave
                with wave.open(decoded_wav, "rb") as wavesrc:
                    frames = wavesrc.readframes(wavesrc.getnframes())
                    pcm16_bytes = frames
                    sr = wavesrc.getframerate()

        # At this point we have pcm16_bytes (PCM16 LE bytes) and sr and video_time
        # Call your user-provided function which returns RELATIVE segments (seconds)
        # Note: your function signature earlier was:
        #   process_audio_bytes(pcm16_bytes: bytes, sample_rate: int, chunk_start_time: float) -> list[dict]
        rel_segments = process_audio_bytes(pcm16_bytes, sr, float(video_time))

        # Map relative segments to absolute timestamps
        abs_segments = []
        for seg in rel_segments:
            label = seg.get("label", "unknown")
            rel_start = float(seg.get("start", 0.0))
            rel_end = float(seg.get("end", 0.0))
            score = float(seg.get("score", 0.0))
            abs_segments.append({
                "label": label,
                "start": float(video_time + rel_start),
                "end": float(video_time + rel_end),
                "score": score
            })

        return {"labels": abs_segments, "frame_start": float(video_time)}
    except HTTPException:
        # re-raise FastAPI HTTP errors unchanged
        raise
    except Exception as e:
        # Helpful debug info (local dev). If you want less detail, reduce message.
        raise HTTPException(status_code=500, detail=f"Internal server error: {type(e).__name__}: {str(e)}")
    finally:
        if tmp_dir:
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


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8887, log_level="debug", access_log=True, use_colors=True)