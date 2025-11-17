#!/usr/bin/env python3
"""
Test client for WhisperX API

This script demonstrates how to use the WhisperX API endpoints in both async and sync modes.

Usage (Async mode with URL):
    python test_whisperx_api.py --base-url <URL> --audio-url "https://example.com/audio.mp3"
    python test_whisperx_api.py --base-url <URL> --audio-url "https://example.com/audio.mp3" --diarize

Usage (Sync mode with local WAV file):
    python test_whisperx_api.py --base-url <URL> --wav-file audio.wav
    python test_whisperx_api.py --base-url <URL> --wav-file audio.wav --diarize --language en

Dependencies:
    pip install requests soundfile librosa
"""

import requests
import time
import json
import argparse
import base64
import numpy as np
import io
from typing import Optional


class WhisperXClient:
    """Client for WhisperX API"""

    def __init__(self, base_url: str):
        """
        Initialize client

        Args:
            base_url: Base URL of the deployed Modal app
                     Example: "https://username--whisperx-api-transcribes.modal.run"
        """
        self.base_url = base_url.rstrip("/")

    def submit_job(
        self,
        audio_url: str,
        language: Optional[str] = None,
        batch_size: int = 16,
        align: bool = True,
        diarize: bool = False,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> dict:
        """
        Submit a transcription job

        Returns:
            dict with job_id and status
        """
        payload = {
            "audio_url": audio_url,
            "batch_size": batch_size,
            "align": align,
            "diarize": diarize,
        }

        if language:
            payload["language"] = language
        if min_speakers is not None:
            payload["min_speakers"] = min_speakers
        if max_speakers is not None:
            payload["max_speakers"] = max_speakers

        response = requests.post(
            f"{self.base_url}/transcribes",
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        return response.json()

    def get_status(self, job_id: str) -> dict:
        """
        Get job status (and result if completed)

        Returns:
            dict with status information and result if completed
        """
        response = requests.get(
            f"{self.base_url}/transcribes",
            params={"job_id": job_id}
        )
        response.raise_for_status()
        return response.json()

    def get_result(self, job_id: str) -> dict:
        """
        Get transcription result (same as get_status, but only returns if completed)

        Returns:
            dict with transcription data
        """
        status = self.get_status(job_id)
        if status["status"] != "completed":
            raise RuntimeError(f"Job not completed yet. Current status: {status['status']}")
        return status

    def delete_job(self, job_id: str) -> dict:
        """
        Delete job from store

        Note: This endpoint may not be implemented on the server.
        """
        try:
            response = requests.delete(
                f"{self.base_url}/transcribes",
                params={"job_id": job_id}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404 or e.response.status_code == 405:
                # Delete endpoint not implemented - this is ok
                return {"status": "delete_not_supported", "job_id": job_id}
            raise

    def wait_for_completion(
        self,
        job_id: str,
        poll_interval: int = 2,
        timeout: int = 3600,
        verbose: bool = True,
    ) -> dict:
        """
        Poll job status until completion

        Args:
            job_id: Job ID to poll
            poll_interval: Seconds between status checks
            timeout: Maximum seconds to wait
            verbose: Print status updates

        Returns:
            Final job status dict

        Raises:
            TimeoutError: If job doesn't complete within timeout
            RuntimeError: If job fails
        """
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")

            status = self.get_status(job_id)

            if verbose:
                progress = status.get("progress", 0)
                message = status.get("message", "")
                print(f"[{elapsed:.1f}s] Status: {status['status']} ({progress}%) - {message}")

            if status["status"] == "completed":
                return status

            if status["status"] == "failed":
                error_msg = status.get("error", "Unknown error")
                raise RuntimeError(f"Job {job_id} failed: {error_msg}")

            time.sleep(poll_interval)

    def transcribe_and_wait(
        self,
        audio_url: str,
        language: Optional[str] = None,
        diarize: bool = False,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        verbose: bool = True,
    ) -> dict:
        """
        Submit job and wait for completion (convenience method)

        Returns:
            Transcription result dict
        """
        # Submit job
        if verbose:
            print(f"Submitting transcription job...")
            print(f"  Audio URL: {audio_url}")
            print(f"  Language: {language or 'auto-detect'}")
            print(f"  Diarization: {diarize}")
            print()

        submission = self.submit_job(
            audio_url=audio_url,
            language=language,
            diarize=diarize,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

        job_id = submission["job_id"]
        if verbose:
            print(f"Job submitted! Job ID: {job_id}")
            print()

        # Wait for completion (this also returns the result when done)
        result = self.wait_for_completion(job_id, verbose=verbose)

        if verbose:
            print()
            print("Done!")
            print()

        return result

    def transcribe_wav_sync(
        self,
        wav_file_path: str,
        language: Optional[str] = None,
        batch_size: int = 16,
        align: bool = True,
        diarize: bool = False,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        session_id: Optional[str] = None,
        verbose: bool = True,
    ) -> dict:
        """
        Synchronously transcribe a WAV file by sending audio samples

        Args:
            wav_file_path: Path to WAV file
            language: Language code (e.g., 'en', 'de')
            batch_size: Batch size for transcription
            align: Enable word-level alignment
            diarize: Enable speaker diarization
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers
            session_id: Session ID for container affinity (optional)
            verbose: Print progress

        Returns:
            Transcription result dict
        """
        import soundfile as sf

        if verbose:
            print(f"Loading audio from: {wav_file_path}")

        # Load audio file as numpy array
        audio, sample_rate = sf.read(wav_file_path)

        if verbose:
            print(f"  Sample rate: {sample_rate} Hz")
            print(f"  Duration: {len(audio) / sample_rate:.2f} seconds")
            print(f"  Samples: {len(audio)}")
            print()

        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
            if verbose:
                print("  Converted stereo to mono")

        # Resample to 16kHz if needed (WhisperX expects 16kHz)
        if sample_rate != 16000:
            if verbose:
                print(f"  Resampling from {sample_rate} Hz to 16000 Hz...")
            import librosa
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000

        # Convert to float32 (expected by WhisperX)
        audio = audio.astype(np.float32)

        # Serialize numpy array to .npy format in memory
        buffer = io.BytesIO()
        np.save(buffer, audio)
        audio_bytes = buffer.getvalue()

        # Encode as base64
        audio_chunk_b64 = base64.b64encode(audio_bytes).decode('utf-8')

        if verbose:
            print(f"Sending {len(audio_bytes)} bytes to server...")
            print()

        # Send to API
        payload = {
            "audio_chunk": audio_chunk_b64,
            "language": language,
            "batch_size": batch_size,
            "align": align,
            "diarize": diarize,
        }

        if min_speakers is not None:
            payload["min_speakers"] = min_speakers
        if max_speakers is not None:
            payload["max_speakers"] = max_speakers
        if session_id is not None:
            payload["session_id"] = session_id

        start_time = time.time()

        # Call the following 360 times to simulate one hour processing
        print(f"Calling the API {360} times to simulate one hour processing: start_time: {time.time()}")
        response = None
        for i in range(360):
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/transcribes",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            end_time = time.time()
            if end_time - start_time > 10:
                print(f"API call {i} took {end_time - start_time} seconds")
            else:
                print(f"API call {i} took {end_time - start_time} seconds")
                time.sleep(10 - (end_time - start_time))
        print(f"Calling the API {360} times to simulate one hour processing: end_time: {time.time()}")
        response.raise_for_status()
        result = response.json()

        elapsed = time.time() - start_time

        if verbose:
            print(f"Transcription completed in {elapsed:.2f} seconds")
            if "session_id" in result:
                print(f"Session ID: {result['session_id']}")
            print()

        return result


def main():
    parser = argparse.ArgumentParser(
        description="Test client for WhisperX API"
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="Base URL of deployed Modal app (e.g., https://username--whisperx-api-transcribes.modal.run)",
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--audio-url",
        help="URL of audio file to transcribe (async mode)",
    )
    mode_group.add_argument(
        "--wav-file",
        help="Path to local WAV file to transcribe (sync mode)",
    )

    parser.add_argument(
        "--language",
        help="Language code (e.g., 'en', 'de', 'fr'). Auto-detect if not provided.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for transcription (default: 16)",
    )
    parser.add_argument(
        "--no-align",
        action="store_true",
        help="Disable word-level alignment",
    )
    parser.add_argument(
        "--diarize",
        action="store_true",
        help="Enable speaker diarization",
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        help="Minimum number of speakers (for diarization)",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        help="Maximum number of speakers (for diarization)",
    )
    parser.add_argument(
        "--session-id",
        help="Session ID for container affinity (sync mode only)",
    )
    parser.add_argument(
        "--output",
        help="Output file path for JSON results",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Create client
    client = WhisperXClient(args.base_url)

    # Transcribe based on mode
    if args.wav_file:
        # Synchronous mode with WAV file
        result = client.transcribe_wav_sync(
            wav_file_path=args.wav_file,
            language=args.language,
            batch_size=args.batch_size,
            align=not args.no_align,
            diarize=args.diarize,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            session_id=args.session_id,
            verbose=not args.quiet,
        )
    else:
        # Async mode with URL
        result = client.transcribe_and_wait(
            audio_url=args.audio_url,
            language=args.language,
            diarize=args.diarize,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            verbose=not args.quiet,
        )

    # Print results
    print("="*80)
    print("TRANSCRIPTION RESULTS")
    print("="*80)
    print(f"Language: {result['language']}")
    print(f"Segments: {len(result['segments'])}")
    print()

    # Print transcript
    print("TRANSCRIPT:")
    print("-"*80)
    for segment in result["segments"]:
        start = segment.get("start", 0)
        end = segment.get("end", 0)
        text = segment.get("text", "")
        speaker = segment.get("speaker", "")

        if args.diarize and speaker:
            print(f"[{start:6.2f}s -> {end:6.2f}s] {speaker}: {text}")
        else:
            print(f"[{start:6.2f}s -> {end:6.2f}s] {text}")

    # Show word-level info if available
    if result.get("word_segments"):
        print()
        print(f"Word-level timestamps: {len(result['word_segments'])} words")
        print("First 10 words:")
        for word_seg in result["word_segments"][:10]:
            word = word_seg.get("word", "")
            start = word_seg.get("start", 0)
            end = word_seg.get("end", 0)
            score = word_seg.get("score", 0)
            speaker = word_seg.get("speaker", "")

            if speaker:
                print(f"  {start:6.2f}s: '{word}' (score: {score:.3f}, speaker: {speaker})")
            else:
                print(f"  {start:6.2f}s: '{word}' (score: {score:.3f})")

    # Save to file if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print()
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
