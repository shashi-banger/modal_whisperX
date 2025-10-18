"""
WhisperX on Modal - Process local audio files

This script uploads local audio files to Modal and processes them with WhisperX.

Usage:
    modal run modal_whisperx_local.py --audio-file "./audio.mp3"
    modal run modal_whisperx_local.py --audio-file "./audio.mp3" --diarize
"""

import modal
import os
from pathlib import Path

# Use the same image as modal_whisperx.py
cuda_version = "12.4.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

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
    .pip_install("whisperx", "ffmpeg-python")
    # Set LD_LIBRARY_PATH to include cuDNN library location
    .env({
        "LD_LIBRARY_PATH": "/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}"
    })
)

app = modal.App("whisperx-local-transcription", image=image)

GPU_CONFIG = "A100-40GB"
CACHE_DIR = "/cache"
cache_vol = modal.Volume.from_name("whisperx-cache", create_if_missing=True)


@app.cls(
    gpu=GPU_CONFIG,
    volumes={CACHE_DIR: cache_vol},
    scaledown_window=60 * 10,
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class WhisperXModel:
    """WhisperX model class for processing uploaded files"""

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
    def transcribe_bytes(
        self,
        audio_data: bytes,
        filename: str,
        language: str = None,
        batch_size: int = 16,
        align: bool = True,
        diarize: bool = False,
        min_speakers: int = None,
        max_speakers: int = None,
    ):
        """
        Transcribe audio from bytes with optional alignment and diarization

        Args:
            audio_data: Raw audio file bytes
            filename: Original filename (for extension detection)
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
        import tempfile
        import gc
        import torch

        # Save to temporary file
        suffix = Path(filename).suffix or ".audio"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
            tmp_file.write(audio_data)
            audio_path = tmp_file.name

        try:
            # Load audio
            print(f"Loading audio from {filename}...")
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

                del align_model
                gc.collect()
                torch.cuda.empty_cache()

            # 3. Assign speaker labels (optional)
            if diarize:
                print("Performing speaker diarization...")

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

                del diarize_model
                gc.collect()
                torch.cuda.empty_cache()

            print("Transcription complete!")
            return {
                "filename": filename,
                "language": detected_language,
                "segments": result["segments"],
                "word_segments": result.get("word_segments", []),
            }

        finally:
            os.unlink(audio_path)


@app.local_entrypoint()
def main(
    audio_file: str,
    language: str = None,
    batch_size: int = 16,
    align: bool = True,
    diarize: bool = False,
    min_speakers: int = None,
    max_speakers: int = None,
    output_file: str = None,
):
    """
    Local entrypoint for processing local audio files

    Example:
        modal run modal_whisperx_local.py --audio-file "./audio.mp3"
        modal run modal_whisperx_local.py --audio-file "./audio.mp3" --diarize --output-file "result.json"
    """
    import json

    # Read local file
    audio_path = Path(audio_file)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    print(f"Reading local file: {audio_file}")
    print(f"File size: {audio_path.stat().st_size / (1024*1024):.2f} MB")

    with open(audio_path, "rb") as f:
        audio_data = f.read()

    print(f"\nStarting WhisperX transcription...")
    print(f"Language: {language or 'auto-detect'}")
    print(f"Alignment: {align}")
    print(f"Diarization: {diarize}")
    print()

    # Process on Modal
    model = WhisperXModel()
    result = model.transcribe_bytes.remote(
        audio_data=audio_data,
        filename=audio_path.name,
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
    print(f"File: {result['filename']}")
    print(f"Language: {result['language']}")
    print(f"Segments: {len(result['segments'])}")
    print("\n")

    # Print full transcript
    print("FULL TRANSCRIPT:")
    print("-"*80)
    for segment in result["segments"]:
        start = segment.get("start", 0)
        end = segment.get("end", 0)
        text = segment.get("text", "")
        speaker = segment.get("speaker", "")

        if diarize and speaker:
            print(f"[{start:6.2f}s -> {end:6.2f}s] {speaker}: {text}")
        else:
            print(f"[{start:6.2f}s -> {end:6.2f}s] {text}")

    # Save to file if requested
    if output_file:
        output_path = Path(output_file)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n{'='*80}")
        print(f"Full results saved to: {output_path.absolute()}")

        # Also save as text file
        txt_path = output_path.with_suffix(".txt")
        with open(txt_path, "w") as f:
            for segment in result["segments"]:
                text = segment.get("text", "").strip()
                if text:
                    f.write(text + " ")
        print(f"Plain text saved to: {txt_path.absolute()}")

    return result
