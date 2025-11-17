"""
WhisperX API on Modal - Simplified transcription service

This creates a REST API with:
- POST /transcribes - Submit transcription (async for URL, sync for audio chunk)

Usage:
    modal deploy modal_whisperx_api.py

Then (Option 1 - Async processing with URL):
    curl -X POST https://your-username--whisperx-api-transcribes.modal.run \
      -H "Content-Type: application/json" \
      -d '{"audio_url": "https://example.com/audio.mp3", "diarize": true}'
    # Returns: {"job_id": "...", "status": "queued"}

    # Poll the same endpoint with job_id to get status/results:
    curl https://your-username--whisperx-api-transcribes.modal.run?job_id=...

Or (Option 2 - Synchronous processing with audio chunk):
    # First chunk - creates session (audio_chunk is base64-encoded numpy array)
    curl -X POST https://your-username--whisperx-api-transcribes.modal.run \
      -H "Content-Type: application/json" \
      -d '{"audio_chunk": "<base64-encoded-numpy-array>"}'
    # Returns immediately: {"status": "completed", "session_id": "...", "segments": [...]}

    # Subsequent chunks - reuse warm container
    curl -X POST https://your-username--whisperx-api-transcribes.modal.run \
      -H "Content-Type: application/json" \
      -d '{"audio_chunk": "<base64-encoded-numpy-array>", "session_id": "..."}'
    # Returns immediately: {"status": "completed", "session_id": "...", "segments": [...]}
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
GPU_CONFIG = "T4"

# Persistent volumes
CACHE_DIR = "/cache"
cache_vol = modal.Volume.from_name("whisperx-cache", create_if_missing=True)

# Dict to store job results (in-memory, could use Modal Dict for persistence)
job_store = modal.Dict.from_name("whisperx-jobs", create_if_missing=True)


@app.cls(
    gpu=GPU_CONFIG,
    volumes={CACHE_DIR: cache_vol},
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("sb-huggingface-secret")],
    allow_concurrent_inputs=10,
    #scaledown_window=0,  # keep warm for 10 minutes after last request
    min_containers=0,  # Process multiple jobs concurrently
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
            "large-v3",
            self.device,
            compute_type=self.compute_type,
            download_root=CACHE_DIR,
        )

        print("WhisperX ASR model loaded successfully")

    @modal.method()
    def process_url_async(
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
        Process a transcription job asynchronously from URL

        This method updates job status in the job_store throughout processing.
        """
        print(f"Processing job {job_id} from URL: {audio_url}")
        import whisperx
        import requests
        import tempfile
        import gc
        import torch
        import traceback

        try:
            # Update status: downloading
            job_store[job_id] = {
                "status": "downloading",
                "progress": 10,
                "message": "Downloading audio file...",
                "updated_at": datetime.utcnow().isoformat(),
            }

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

                # 2. Align (if requested and segments exist)
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

                # 3. Diarize (if requested)
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

    @modal.method()
    def transcribe_buffer(
        self,
        audio_bytes: bytes,
        language: Optional[str] = None,
        batch_size: int = 16,
        align: bool = True,
        diarize: bool = False,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ):
        """
        Synchronous transcription for audio buffer (numpy array in .npy format).

        This method processes audio immediately and returns results.
        Designed for real-time/streaming scenarios with session persistence.
        """
        print(f"Processing buffer transcription, asr_model is: {self.asr_model}")
        import whisperx
        import gc
        import torch
        import traceback
        import numpy as np
        import io

        try:
            # Deserialize numpy array from .npy format
            audio = np.load(io.BytesIO(audio_bytes))

            # 1. Transcribe
            result = self.asr_model.transcribe(
                audio,
                batch_size=batch_size,
                language=language,
            )

            detected_language = result.get("language", "en")

            # 2. Align (if requested and segments exist)
            if align and len(result["segments"]) > 0:
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

            # 3. Diarize (if requested)
            if diarize:
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

            # Return result immediately
            return {
                "status": "completed",
                "language": detected_language,
                "segments": result["segments"],
                "word_segments": result.get("word_segments", []),
            }

        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            print(f"Buffer transcription failed: {error_msg}")
            print(stack_trace)

            return {
                "status": "failed",
                "error": error_msg,
                "stack_trace": stack_trace,
            }


# Expose the FastAPI app via Modal ASGI
@app.function(
    image=image,
    secrets=[modal.Secret.from_name("sb-huggingface-secret")],
)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import base64
    import io
    import numpy as np

    web_app = FastAPI()

    class TranscriptionRequest(BaseModel):
        audio_url: Optional[str] = None
        audio_chunk: Optional[str] = None  # base64-encoded numpy array
        language: Optional[str] = None
        batch_size: int = 16
        align: bool = True
        diarize: bool = False
        min_speakers: Optional[int] = None
        max_speakers: Optional[int] = None
        session_id: Optional[str] = None

    @web_app.post("/transcribes")
    async def transcribe(request: TranscriptionRequest):
        """
        Unified transcription endpoint:
        - audio_chunk provided → synchronous processing, returns results immediately
        - audio_url provided → async processing with job queue
        """
        import uuid

        # Decode audio chunk if provided
        audio_bytes = None
        if request.audio_chunk:
            try:
                # Decode base64 string to bytes
                audio_bytes = base64.b64decode(request.audio_chunk)
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to decode base64 audio_chunk: {str(e)}"
                )

        # Validate that either audio_url or audio_chunk is provided
        if not request.audio_url and not audio_bytes:
            raise HTTPException(
                status_code=400,
                detail="Either audio_url or audio_chunk must be provided"
            )

        # Both provided is ambiguous
        if request.audio_url and audio_bytes:
            raise HTTPException(
                status_code=400,
                detail="Provide either audio_url OR audio_chunk, not both"
            )

        # Handle audio_chunk: synchronous processing with session-based routing
        if audio_bytes:
            # Use session_id for container affinity if provided, otherwise create one
            if not request.session_id:
                session_id = str(uuid.uuid4())
            else:
                session_id = request.session_id

            # Call synchronous transcribe_buffer method
            # Modal's scaledown_window keeps container warm for subsequent requests
            worker = WhisperXWorker()
            result = worker.transcribe_buffer.remote(
                audio_bytes=audio_bytes,
                language=request.language,
                batch_size=request.batch_size,
                align=request.align,
                diarize=request.diarize,
                min_speakers=request.min_speakers,
                max_speakers=request.max_speakers,
            )

            # Return result immediately with session_id for subsequent requests
            result["session_id"] = session_id
            return result

        # Handle audio_url: async processing with job queue
        else:
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

            # Spawn async processing job
            WhisperXWorker().process_url_async.spawn(
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
                "message": "Job submitted successfully. Poll this endpoint with ?job_id=<id> to get status",
            }

    @web_app.get("/transcribes")
    def get_job_status(job_id: str):
        """Poll for job status/results"""
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

    @web_app.get("/health")
    def health():
        return {
            "status": "healthy",
            "service": "whisperx-api",
            "timestamp": datetime.utcnow().isoformat(),
        }

    return web_app
