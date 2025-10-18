# WhisperX on Modal - Complete Guide

This directory contains everything needed to run WhisperX on Modal's serverless GPU infrastructure.

## 📁 Files Overview

### Core Scripts

| File | Purpose | Usage |
|------|---------|-------|
| `modal_whisperx.py` | Process audio from URLs | `modal run modal_whisperx.py --audio-url "https://..."` |
| `modal_whisperx_local.py` | Process local audio files | `modal run modal_whisperx_local.py --audio-file "./audio.mp3"` |
| `modal_whisperx_api.py` | Async REST API with job queue | `modal deploy modal_whisperx_api.py` |
| `test_whisperx_api.py` | Python client for the API | `python test_whisperx_api.py --base-url "https://..." --audio-url "https://..."` |

### Documentation

| File | Description |
|------|-------------|
| `MODAL_SETUP.md` | Installation, setup, and usage guide |
| `API_DOCUMENTATION.md` | Complete REST API documentation with examples |
| `MODAL_CUDNN_FIX.md` | cuDNN library troubleshooting guide |
| `CLAUDE.md` | Architecture overview for AI assistants |

## 🚀 Quick Start

### 1. Setup Modal

```bash
# Install Modal
pip install modal

# Authenticate
modal setup

# (Optional) Setup HuggingFace token for diarization
modal secret create huggingface-secret HF_TOKEN=your_token_here
```

### 2. Run WhisperX

**Process URL:**
```bash
modal run modal_whisperx.py \
  --audio-url "https://example.com/audio.mp3" \
  --diarize
```

**Process local file:**
```bash
modal run modal_whisperx_local.py \
  --audio-file "./my_audio.mp3" \
  --diarize \
  --output-file "transcript.json"
```

**Deploy as API:**
```bash
modal deploy modal_whisperx_api.py

# Then use the API
curl -X POST https://{your-username}--whisperx-api-fastapi-app.modal.run/transcribes \                                              
    -H 'Content-Type: application/json' \
    -d '{"audio_url": "https://d3pw1gn2woo883.cloudfront.net/tmp/troy_trunc_15m.mp3", "diarize": true}'
```

## ⚙️ Configuration

All scripts support:

- **Language**: `--language en` (auto-detect if omitted)
- **Batch size**: `--batch-size 16` (increase for faster processing)
- **Alignment**: `--align` (enabled by default for word-level timestamps)
- **Diarization**: `--diarize` (speaker identification)
- **Speaker hints**: `--min-speakers 2 --max-speakers 4`

## 🔧 cuDNN Fix (Important!)

All scripts include the fix for the cuDNN library loading issue:

```python
.pip_install("nvidia-cudnn-cu12==9.8.0.87")
.env({
    "LD_LIBRARY_PATH": "/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}"
})
```

This ensures PyTorch 2.8.0 can find the correct cuDNN libraries. See `MODAL_CUDNN_FIX.md` for details.

## 📊 Features

### Full WhisperX Pipeline

1. **ASR (Automatic Speech Recognition)**
   - Uses OpenAI's Whisper model (large-v2)
   - 70x realtime speed on A100
   - Automatic language detection

2. **Word-level Alignment**
   - Wav2vec2-based forced alignment
   - Accurate timestamps per word
   - Confidence scores included

3. **Speaker Diarization**
   - PyAnnote-audio based speaker identification
   - Labels speakers as SPEAKER_00, SPEAKER_01, etc.
   - Optional speaker count hints

### API Features

- **Async job processing** with job IDs
- **Status polling** with progress updates (0-100%)
- **Concurrent processing** (up to 10 jobs)
- **Persistent storage** using Modal Dict
- **Error handling** with detailed stack traces

## 📈 Performance

On A100 GPU:

| Audio Length | ASR Only | + Alignment | + Diarization |
|--------------|----------|-------------|---------------|
| 1 minute     | ~1s      | ~2s         | ~3s           |
| 10 minutes   | ~8s      | ~15s        | ~25s          |
| 1 hour       | ~50s     | ~90s        | ~150s         |

**Note**: First run adds ~30s for model downloads (cached thereafter)

## 💰 Cost Estimate

Modal pricing (A100):
- **GPU**: ~$1.10/hour
- **Storage**: Free up to 50GB
- **Compute**: Billed per second

**Example**: 100 hours of audio with diarization
- Processing time: ~4 hours
- Cost: **~$4.40**

## 🎯 Output Format

```json
{
  "language": "en",
  "segments": [
    {
      "start": 0.0,
      "end": 3.5,
      "text": "Hello and welcome to this podcast.",
      "speaker": "SPEAKER_00",
      "words": [
        {
          "word": "Hello",
          "start": 0.0,
          "end": 0.5,
          "score": 0.95,
          "speaker": "SPEAKER_00"
        }
      ]
    }
  ],
  "word_segments": [...]
}
```

## 🐛 Troubleshooting

### Common Issues

**cuDNN errors**: See `MODAL_CUDNN_FIX.md`

**HF token errors**: Setup secret:
```bash
modal secret create huggingface-secret HF_TOKEN=your_token
```

**Out of memory**: Reduce batch size or use smaller model

**Slow first run**: Models are downloading (~5GB)

### Check Logs

```bash
# View app logs
modal app logs whisperx-transcription

# Stop and rebuild
modal app stop whisperx-transcription
```

## 📚 Advanced Usage

### Customize Model Size

Edit the scripts to change model:

```python
self.asr_model = whisperx.load_model(
    "medium",  # or "small", "large-v3"
    self.device,
    compute_type=self.compute_type,
    download_root=CACHE_DIR,
)
```

### Change GPU Type

```python
GPU_CONFIG = modal.gpu.H100(count=1)  # or .L4(), .T4(), .A100()
```

### Deploy as Web Service

The API can be deployed with `keep_warm` to maintain hot containers:

```python
@app.cls(
    gpu=GPU_CONFIG,
    keep_warm=1,  # Keep 1 container always running
    ...
)
```

## 🔗 Links

- [WhisperX GitHub](https://github.com/m-bain/whisperX)
- [Modal Documentation](https://modal.com/docs)
- [PyAnnote Audio](https://github.com/pyannote/pyannote-audio)
- [Whisper Paper](https://arxiv.org/abs/2303.00747)

## 📝 License

WhisperX is licensed under BSD-2-Clause. See the main repository for details.

---

**Questions?** Check the documentation files or Modal's support at https://modal.com/docs
