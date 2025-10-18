"""
WhisperX on Modal - Fast ASR with alignment and diarization

This script runs WhisperX on Modal's serverless infrastructure with GPU acceleration.

Usage:
    modal run modal_whisperx.py --audio-url "https://example.com/audio.mp3"
    modal run modal_whisperx.py --audio-url "https://example.com/audio.mp3" --diarize --hf-token "your_token"
"""

import modal
import os

# CUDA configuration for WhisperX
cuda_version = "12.4.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

# Build custom image with WhisperX dependencies
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
    .pip_install(
        "whisperx",  # Install latest stable version
        "ffmpeg-python",
    )
    # Set LD_LIBRARY_PATH to include cuDNN library location
    .env({
        "LD_LIBRARY_PATH": "/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}"
    })
)

app = modal.App("whisperx-transcription", image=image)

# GPU configuration - use A100 or H100 for best performance
GPU_CONFIG = "A100-40GB"

# Persistent volume for model caching
CACHE_DIR = "/cache"
cache_vol = modal.Volume.from_name("whisperx-cache", create_if_missing=True)


@app.cls(
    gpu="A100-40GB",
    volumes={CACHE_DIR: cache_vol},
    scaledown_window=60 * 10,  # Keep warm for 10 minutes
    timeout=60 * 60,  # 1 hour timeout
    secrets=[modal.Secret.from_name("huggingface-secret")],  # For diarization
)
class WhisperXModel:
    """WhisperX model class with full pipeline support"""

    @modal.enter()
    def setup(self):
        """Load models on container startup"""
        import whisperx
        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16"

        print(f"Loading WhisperX model on {self.device}...")

        # Load ASR model
        self.asr_model = whisperx.load_model(
            "large-v3",
            self.device,
            compute_type=self.compute_type,
            download_root=CACHE_DIR,
        )

        print("WhisperX ASR model loaded successfully")

    @modal.method()
    def transcribe(
        self,
        audio_url: str,
        language: str = None,
        batch_size: int = 16,
        align: bool = True,
        diarize: bool = False,
        min_speakers: int = None,
        max_speakers: int = None,
    ):
        """
        Transcribe audio with optional alignment and diarization

        Args:
            audio_url: URL to audio file
            language: Language code (e.g., 'en', 'de', 'fr'). Auto-detect if None.
            batch_size: Batch size for inference
            align: Whether to perform word-level alignment
            diarize: Whether to perform speaker diarization
            min_speakers: Minimum number of speakers (for diarization)
            max_speakers: Maximum number of speakers (for diarization)

        Returns:
            Dictionary with transcription results
        """
        import whisperx
        import requests
        import tempfile
        import gc
        import torch

        # Download audio file
        print(f"Downloading audio from {audio_url}...")
        response = requests.get(audio_url)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as tmp_file:
            tmp_file.write(response.content)
            audio_path = tmp_file.name

        try:
            # Load audio
            print("Loading audio...")
            audio = whisperx.load_audio(audio_path)

            # 1. Transcribe with Whisper
            print("Transcribing...")
            result = self.asr_model.transcribe(
                audio,
                batch_size=batch_size,
                language=language,
            )

            detected_language = result.get("language", "en")
            print(f"Detected language: {detected_language}")

            # 2. Align whisper output (word-level timestamps)
            if align and len(result["segments"]) > 0:
                print("Performing alignment...")
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

                # Free alignment model memory
                del align_model
                gc.collect()
                torch.cuda.empty_cache()

            # 3. Assign speaker labels (optional)
            if diarize:
                print("Performing speaker diarization...")

                # Get HuggingFace token from environment
                hf_token = os.environ.get("HF_TOKEN")
                if not hf_token:
                    print("Warning: HF_TOKEN not found, diarization may fail")

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

                # Free diarization model memory
                del diarize_model
                gc.collect()
                torch.cuda.empty_cache()

            print("Transcription complete!")
            return {
                "language": detected_language,
                "segments": result["segments"],
                "word_segments": result.get("word_segments", []),
            }

        finally:
            # Clean up temp file
            os.unlink(audio_path)


@app.local_entrypoint()
def main(
    audio_url: str,
    language: str = None,
    batch_size: int = 16,
    align: bool = True,
    diarize: bool = False,
    min_speakers: int = None,
    max_speakers: int = None,
    output_file: str = None,
):
    """
    Local entrypoint for running WhisperX transcription

    Example:
        modal run modal_whisperx.py --audio-url "https://example.com/audio.mp3"
        modal run modal_whisperx.py --audio-url "https://example.com/audio.mp3" --diarize
    """
    import json

    print(f"Starting WhisperX transcription...")
    print(f"Audio URL: {audio_url}")
    print(f"Language: {language or 'auto-detect'}")
    print(f"Alignment: {align}")
    print(f"Diarization: {diarize}")

    model = WhisperXModel()
    result = model.transcribe.remote(
        audio_url=audio_url,
        language=language,
        batch_size=batch_size,
        align=align,
        diarize=diarize,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    # Print results
    print("\n" + "="*80)
    print("TRANSCRIPTION RESULTS")
    print("="*80)
    print(f"Language: {result['language']}")
    print(f"Segments: {len(result['segments'])}")
    print("\n")

    for i, segment in enumerate(result["segments"][:5]):  # Show first 5 segments
        start = segment.get("start", "N/A")
        end = segment.get("end", "N/A")
        text = segment.get("text", "")
        speaker = segment.get("speaker", "N/A")

        print(f"[{start:.2f}s -> {end:.2f}s] Speaker {speaker}: {text}")

    if len(result["segments"]) > 5:
        print(f"\n... and {len(result['segments']) - 5} more segments")

    # Save to file if requested
    if output_file:
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nFull results saved to: {output_file}")

    return result
