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

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
# server.py
"""
FastAPI server for HumSkip prototype.

Endpoints:
  POST /infer_video    -> start processing a YouTube URL (returns job_id)
  GET  /jobs/{job_id}/status -> get job progress
  GET  /jobs/{job_id}/result -> get timeline JSON (once done)

Processing is done in background threads via run_in_executor to avoid blocking the event loop.
This design is resilient for long-running CPU-bound jobs.

Assumptions:
- You provide `process_audio_from_video(video_url, out_json_path)` somewhere on PYTHONPATH.
  If not present, a dummy processor will create a trivial timeline (no hums).
"""

import asyncio
import concurrent.futures
import json
import os
import pathlib
import shutil
import tempfile
import time
import uuid
import video_utils
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="HumSkip Local Inference Server (Prototype)")

# allow local pages to access this API (Tampermonkey uses GM_xmlhttpRequest but CORS is still reasonable)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://127.0.0.1", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory job store (for prototype). For production replace with Redis / DB.
JOB_STORE: Dict[str, Dict] = {}
JOB_LOCK = asyncio.Lock()

# Executor for CPU bound jobs
EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=2)

# Directory to store outputs
BASE_OUT_DIR = pathlib.Path.cwd() / "humskip_jobs"
BASE_OUT_DIR.mkdir(parents=True, exist_ok=True)


class InferRequest(BaseModel):
    url: str
    video_id: Optional[str] = None


def _dummy_processor(video_url: str, out_json_path: str):
    """
    Simple fallback: sleep a few seconds and write a trivial JSON with no hums.
    This keeps the prototype functional if user function is absent.
    """
    print(f"[dummy] processing {video_url} ...")
    time.sleep(2)
    example = {
        "video_id": "unknown",
        "duration": 60.0,
        "sample_rate": 16000,
        "segments": [
            {"label": "speech", "start": 0.0, "end": 60.0, "score": 0.99}
        ],
    }
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(example, f, indent=2)
    print(f"[dummy] wrote {out_json_path}")
    


def run_processing_sync(video_url: str, video_id: str, out_json_path: str, job_updater=None):
    """
    Synchronous wrapper that runs the actual user processing.
    job_updater: optional callable(progress_dict) to update job progress in JOB_STORE.
    This runs in a background thread via run_in_executor.
    """
    print(f"[five sampling processor] processing {video_url} ...")
    
    wav_path = video_utils.youtube_dl_wav(video_url)
    
    example = {
        "video_id": video_id,
        "duration": 60.0,
        "sample_rate": 16000,
        "segments": [
            {"label": "speech", "start": 0.0, "end": 60.0, "score": 0.99}
        ],
    }
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(example, f, indent=2)
    print(f"[dummy] wrote {out_json_path}")
    



async def _start_job(video_url: str, video_id: Optional[str] = None) -> str:
    """
    Create a job, schedule it on executor, and return job_id.
    """
    job_id = str(uuid.uuid4())
    created = time.time()
    out_dir = BASE_OUT_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = str(out_dir / f"{(video_id or job_id)}_hum_timeline.json")

    job_entry = {
        "job_id": job_id,
        "video_id": video_id or "unknown",
        "status": "queued",
        "created_at": created,
        "started_at": None,
        "finished_at": None,
        "progress": 0.0,
        "out_json": out_json,
        "error": None,
    }
    async with JOB_LOCK:
        JOB_STORE[job_id] = job_entry

    loop = asyncio.get_event_loop()

    # worker wrapper to update job store status safely
    def _worker():
        try:
            # mark running
            JOB_STORE[job_id]["status"] = "running"
            JOB_STORE[job_id]["started_at"] = time.time()
            # run synchronous processing (user-supplied) in thread
            run_processing_sync(video_url, video_id or job_id, out_json)
            # mark done
            JOB_STORE[job_id]["status"] = "done"
            JOB_STORE[job_id]["finished_at"] = time.time()
            JOB_STORE[job_id]["progress"] = 1.0
        except Exception as e:
            JOB_STORE[job_id]["status"] = "failed"
            JOB_STORE[job_id]["error"] = repr(e)
            JOB_STORE[job_id]["finished_at"] = time.time()
            print(f"[server] job {job_id} failed: {e}")

    # schedule on executor (non-blocking)
    loop.run_in_executor(EXECUTOR, _worker)
    return job_id


@app.post("/infer_video")
async def infer_video(req: InferRequest):
    """
    Start processing a video. Request body JSON: {"url": "...", "video_id": "..."}.
    Response: {"job_id": "..."}.
    """
    if not req.url:
        raise HTTPException(status_code=400, detail="`url` required")
    job_id = await _start_job(req.url, req.video_id)
    return {"job_id": job_id}


@app.get("/jobs/{job_id}/status")
async def job_status(job_id: str):
    """
    Return job status. Example:
      { "status": "running", "progress": 0.3, "created_at":..., "started_at":..., "finished_at":..., "error":... }
    """
    async with JOB_LOCK:
        job = JOB_STORE.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        # return minimal info
        return {
            "job_id": job_id,
            "video_id": job.get("video_id"),
            "status": job.get("status"),
            "progress": job.get("progress"),
            "created_at": job.get("created_at"),
            "started_at": job.get("started_at"),
            "finished_at": job.get("finished_at"),
            "error": job.get("error"),
        }


@app.get("/jobs/{job_id}/result")
async def job_result(job_id: str):
    """
    Return the JSON result of the job; 404 if missing; 202 if not done yet.
    """
    async with JOB_LOCK:
        job = JOB_STORE.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        if job["status"] == "failed":
            raise HTTPException(status_code=500, detail=f"job failed: {job.get('error')}")
        if job["status"] != "done":
            # still processing
            return {"status": job["status"], "progress": job.get("progress", 0.0)}
        out_path = job.get("out_json")
    # Serve file content
    if not out_path or not os.path.exists(out_path):
        raise HTTPException(status_code=404, detail="result not found")
    # Read and return JSON
    with open(out_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


# Optional utility endpoint to list jobs (for debugging)
@app.get("/jobs")
async def jobs_list():
    async with JOB_LOCK:
        return {jid: {"status": j["status"], "video_id": j["video_id"]} for jid, j in JOB_STORE.items()}


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("server:app", host="127.0.0.1", port=8887, log_level="info")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8887, log_level="debug", access_log=True, use_colors=True)