# WhisperX on Modal

A complete implementation of [WhisperX](https://github.com/m-bain/whisperX) running on [Modal's](https://modal.com) serverless GPU infrastructure. This project provides high-performance speech-to-text transcription with word-level alignment and speaker diarization capabilities.


## 📁 Project Structure

| File | Purpose |
|------|---------|
| `modal_whisperx.py` | Process audio from URLs |
| `modal_whisperx_local.py` | Process local audio files |
| `modal_whisperx_api.py` | Deploy as REST API service |
| `test_whisperx_api.py` | Python client for the API |

## API Service Deployment

```bash
modal deploy modal_whisperx_api.py
```

## Test the API Service

```bash
# Synchronous mode
python test_whisperx_api.py --base-url https://{your-username}--whisperx-api-fastapi-app.modal.run  --wav-file /home/sbanger/sb/media/whisperx_test_audio/troy_10s.wav
```

```bash
# Async mode
python test_whisperx_api.py --base-url https://{your-username}--whisperx-api-fastapi-app.modal.run  --audio-url https://d3pw1gn2woo883.cloudfront.net/tmp/troy_trunc_15m.mp3
```

## 🛠️ Requirements

- Modal account (free tier available)
- HuggingFace token (for diarization only)
- Audio files in common formats (MP3, WAV, M4A, etc.)

