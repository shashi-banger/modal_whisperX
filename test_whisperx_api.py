#!/usr/bin/env python3
"""
Test client for WhisperX API

This script demonstrates how to use the WhisperX API endpoints.

Usage:
    python test_whisperx_api.py --audio-url "https://example.com/audio.mp3"
    python test_whisperx_api.py --audio-url "https://example.com/audio.mp3" --diarize
"""

import requests
import time
import json
import argparse
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
        response = requests.get(f"{self.base_url}/transcribes/{job_id}")
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
        """Delete job from store"""
        response = requests.delete(f"{self.base_url}/transcribes/{job_id}")
        response.raise_for_status()
        return response.json()

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


def main():
    parser = argparse.ArgumentParser(
        description="Test client for WhisperX API"
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="Base URL of deployed Modal app (e.g., https://username--whisperx-api-transcribes.modal.run)",
    )
    parser.add_argument(
        "--audio-url",
        required=True,
        help="URL of audio file to transcribe",
    )
    parser.add_argument(
        "--language",
        help="Language code (e.g., 'en', 'de', 'fr'). Auto-detect if not provided.",
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

    # Transcribe and wait
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
