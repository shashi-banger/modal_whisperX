"""
WhisperX API on Modal - Async transcription service with job queue

This creates a REST API with:
- POST /transcribes - Submit transcription job, get job_id
- GET /transcribes/{job_id} - Check job status
- GET /result/{job_id} - Get completed transcription

Usage:
    modal deploy modal_whisperx_api.py

Then:
    curl -X POST https://your-username--whisperx-api-transcribes.modal.run \
      -H "Content-Type: application/json" \
      -d '{"audio_url": "https://example.com/audio.mp3", "diarize": true}'
"""

import modal
import os
from typing import Optional
from datetime import datetime
import json

# CUDA configuration
cuda_version = "12.4.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

# Build image
image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "torch==2.8.0",
        "torchaudio==2.8.0",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    # Install specific cuDNN version to fix library loading issues
    .pip_install("nvidia-cudnn-cu12==9.8.0.87")
    .pip_install("whisperx", "ffmpeg-python", "fastapi", "pydantic")
    # Set LD_LIBRARY_PATH to include cuDNN library location
    .env({
        "LD_LIBRARY_PATH": "/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}"
    })
)

app = modal.App("whisperx-api", image=image)

# GPU configuration
GPU_CONFIG = "A100-40GB"

# Persistent volumes
CACHE_DIR = "/cache"
cache_vol = modal.Volume.from_name("whisperx-cache", create_if_missing=True)

# Dict to store job results (in-memory, could use Modal Dict for persistence)
job_store = modal.Dict.from_name("whisperx-jobs", create_if_missing=True)


@app.cls(
    gpu=GPU_CONFIG,
    volumes={CACHE_DIR: cache_vol},
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    allow_concurrent_inputs=10,  # Process multiple jobs concurrently
)
class WhisperXWorker:
    """WhisperX worker that processes transcription jobs"""

    @modal.enter()
    def setup(self):
        """Load models on container startup"""
        import whisperx
        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16"

        print(f"Loading WhisperX model on {self.device}...")

        self.asr_model = whisperx.load_model(
            "large-v2",
            self.device,
            compute_type=self.compute_type,
            download_root=CACHE_DIR,
        )

        print("WhisperX ASR model loaded successfully")

    @modal.method()
    def process_job(
        self,
        job_id: str,
        audio_url: str,
        language: Optional[str] = None,
        batch_size: int = 16,
        align: bool = True,
        diarize: bool = False,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ):
        """
        Process a transcription job asynchronously

        This method updates job status in the job_store throughout processing.
        """
        print(f"Processing job {job_id}, asr_model is: {self.asr_model}")
        import whisperx
        import requests
        import tempfile
        import gc
        import torch
        import traceback

        # Update status: downloading
        job_store[job_id] = {
            "status": "downloading",
            "progress": 10,
            "message": "Downloading audio file...",
            "updated_at": datetime.utcnow().isoformat(),
        }

        try:
            # Download audio file
            response = requests.get(audio_url, timeout=300)
            response.raise_for_status()

            with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as tmp_file:
                tmp_file.write(response.content)
                audio_path = tmp_file.name

            try:
                # Update status: loading audio
                job_store[job_id] = {
                    "status": "processing",
                    "progress": 20,
                    "message": "Loading audio...",
                    "updated_at": datetime.utcnow().isoformat(),
                }

                audio = whisperx.load_audio(audio_path)

                # Update status: transcribing
                job_store[job_id] = {
                    "status": "processing",
                    "progress": 30,
                    "message": "Transcribing with Whisper...",
                    "updated_at": datetime.utcnow().isoformat(),
                }

                # 1. Transcribe
                result = self.asr_model.transcribe(
                    audio,
                    batch_size=batch_size,
                    language=language,
                )

                detected_language = result.get("language", "en")

                # Update status: aligning
                if align and len(result["segments"]) > 0:
                    job_store[job_id] = {
                        "status": "processing",
                        "progress": 60,
                        "message": "Performing word-level alignment...",
                        "updated_at": datetime.utcnow().isoformat(),
                    }

                    align_model, metadata = whisperx.load_align_model(
                        language_code=detected_language,
                        device=self.device,
                    )

                    result = whisperx.align(
                        result["segments"],
                        align_model,
                        metadata,
                        audio,
                        self.device,
                        return_char_alignments=False,
                    )

                    del align_model
                    gc.collect()
                    torch.cuda.empty_cache()

                # Update status: diarizing
                if diarize:
                    job_store[job_id] = {
                        "status": "processing",
                        "progress": 80,
                        "message": "Performing speaker diarization...",
                        "updated_at": datetime.utcnow().isoformat(),
                    }

                    hf_token = os.environ.get("HF_TOKEN")

                    from whisperx.diarize import DiarizationPipeline

                    diarize_model = DiarizationPipeline(
                        use_auth_token=hf_token,
                        device=self.device,
                    )

                    diarize_segments = diarize_model(
                        audio,
                        min_speakers=min_speakers,
                        max_speakers=max_speakers,
                    )

                    result = whisperx.assign_word_speakers(diarize_segments, result)

                    del diarize_model
                    gc.collect()
                    torch.cuda.empty_cache()

                # Update status: complete
                job_store[job_id] = {
                    "status": "completed",
                    "progress": 100,
                    "message": "Transcription complete",
                    "updated_at": datetime.utcnow().isoformat(),
                    "result": {
                        "language": detected_language,
                        "segments": result["segments"],
                        "word_segments": result.get("word_segments", []),
                    },
                }

            finally:
                os.unlink(audio_path)

        except Exception as e:
            # Update status: failed
            error_msg = str(e)
            stack_trace = traceback.format_exc()

            job_store[job_id] = {
                "status": "failed",
                "progress": 0,
                "message": f"Error: {error_msg}",
                "error": error_msg,
                "stack_trace": stack_trace,
                "updated_at": datetime.utcnow().isoformat(),
            }
            print(f"Job {job_id} failed: {error_msg}")
            print(stack_trace)


# Expose the FastAPI app via Modal ASGI
@app.function(
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI
    from pydantic import BaseModel

    web_app = FastAPI()

    class TranscriptionRequest(BaseModel):
        audio_url: str
        language: str = None
        batch_size: int = 16
        align: bool = True
        diarize: bool = False
        min_speakers: int = None
        max_speakers: int = None

    @web_app.post("/transcribes")
    def create_transcription(request: TranscriptionRequest):
        import uuid

        job_id = str(uuid.uuid4())

        # Initialize job status
        job_store[job_id] = {
            "status": "queued",
            "progress": 0,
            "message": "Job queued for processing",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "params": {
                "audio_url": request.audio_url,
                "language": request.language,
                "batch_size": request.batch_size,
                "align": request.align,
                "diarize": request.diarize,
                "min_speakers": request.min_speakers,
                "max_speakers": request.max_speakers,
            }
        }

        # Spawn async processing job using Modal's spawn method
        WhisperXWorker().process_job.spawn(
            job_id=job_id,
            audio_url=request.audio_url,
            language=request.language,
            batch_size=request.batch_size,
            align=request.align,
            diarize=request.diarize,
            min_speakers=request.min_speakers,
            max_speakers=request.max_speakers,
        )

        return {
            "job_id": job_id,
            "status": "queued",
            "message": "Job submitted successfully",
        }

    @web_app.get("/transcribes/{job_id}")
    def get_transcription(job_id: str):
        if job_id not in job_store:
            return {"error": "Job not found", "job_id": job_id}, 404

        job_data = job_store[job_id]

        response = {
            "job_id": job_id,
            "status": job_data["status"],
            "progress": job_data.get("progress", 0),
            "message": job_data.get("message", ""),
            "updated_at": job_data.get("updated_at", ""),
        }

        if job_data["status"] == "completed":
            result = job_data.get("result", {})
            response["language"] = result.get("language", "unknown")
            response["segments"] = result.get("segments", [])
            response["word_segments"] = result.get("word_segments", [])

        if job_data["status"] == "failed":
            response["error"] = job_data.get("error", "Unknown error")

        return response

    @web_app.get("/transcribe-results/{job_id}")
    def get_result(job_id: str):
        if job_id not in job_store:
            return {"error": "Job not found", "job_id": job_id}, 404

        job_data = job_store[job_id]

        if job_data["status"] != "completed":
            return {
                "error": f"Job not completed yet. Current status: {job_data['status']}",
                "job_id": job_id,
                "status": job_data["status"],
                "progress": job_data.get("progress", 0),
                "message": job_data.get("message", ""),
            }, 400

        result = job_data.get("result", {})
        return {
            "job_id": job_id,
            "status": "completed",
            "language": result.get("language", "unknown"),
            "segments": result.get("segments", []),
            "word_segments": result.get("word_segments", []),
        }

    @web_app.delete("/transcribes/{job_id}")
    def delete_transcription(job_id: str):
        if job_id not in job_store:
            return {"error": "Job not found", "job_id": job_id}, 404

        del job_store[job_id]

        return {
            "message": "Job deleted successfully",
            "job_id": job_id,
        }

    @web_app.get("/health")
    def health():
        return {
            "status": "healthy",
            "service": "whisperx-api",
            "timestamp": datetime.utcnow().isoformat(),
        }

    return web_app
