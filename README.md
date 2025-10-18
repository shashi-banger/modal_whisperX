# WhisperX on Modal

A complete implementation of [WhisperX](https://github.com/m-bain/whisperX) running on [Modal's](https://modal.com) serverless GPU infrastructure. This project provides high-performance speech-to-text transcription with word-level alignment and speaker diarization capabilities.

## 🚀 Quick Start

```bash
# Install Modal
pip install modal

# Setup Modal account
modal setup

# Run transcription on a URL
modal run modal_whisperx.py --audio-url "https://example.com/audio.mp3" --diarize
```

## 📚 Documentation

| Document | Description |
|----------|-------------|
| **[MODAL_README.md](MODAL_README.md)** | Complete guide with features, performance metrics, and examples |
| **[MODAL_SETUP.md](MODAL_SETUP.md)** | Installation, setup, and usage instructions |
| **[API_FINAL_DESIGN.md](API_FINAL_DESIGN.md)** | RESTful API documentation and design |
| **[MODAL_CUDNN_FIX.md](MODAL_CUDNN_FIX.md)** | cuDNN library troubleshooting guide |

## 🎯 What You Get

- **70x realtime** speech transcription on A100 GPUs
- **Word-level timestamps** with confidence scores
- **Speaker diarization** (who spoke when)
- **Multiple interfaces**: CLI scripts, REST API, local file processing
- **Cost-effective**: ~$4.40 for 100 hours of audio processing

## 📁 Project Structure

| File | Purpose |
|------|---------|
| `modal_whisperx.py` | Process audio from URLs |
| `modal_whisperx_local.py` | Process local audio files |
| `modal_whisperx_api.py` | Deploy as REST API service |
| `test_whisperx_api.py` | Python client for the API |

## 🔧 Key Features

### Full WhisperX Pipeline
- **ASR**: OpenAI Whisper (large-v2) with automatic language detection
- **Alignment**: Wav2vec2-based word-level timestamps
- **Diarization**: PyAnnote-audio speaker identification

### Multiple Usage Options
- **CLI**: Direct command-line processing
- **API**: RESTful web service with job queue
- **Local**: Process files from your machine

### Production Ready
- **Async processing** with job tracking
- **Error handling** and retry logic
- **Cost optimization** with GPU auto-scaling
- **Persistent caching** for models and results

## 💡 Use Cases

- **Podcast transcription** with speaker identification
- **Meeting recordings** with word-level timestamps
- **Video content** subtitle generation
- **Batch processing** of audio archives
- **Real-time applications** via API

## 🛠️ Requirements

- Modal account (free tier available)
- HuggingFace token (for diarization only)
- Audio files in common formats (MP3, WAV, M4A, etc.)

## 📈 Performance

On A100 GPU:
- **1 hour audio**: ~50 seconds processing
- **10 hours audio**: ~8 minutes processing
- **100 hours audio**: ~1.5 hours processing

## 💰 Pricing

Modal A100 pricing:
- **GPU**: ~$1.10/hour
- **Storage**: Free up to 50GB
- **Example**: 100 hours of audio = ~$4.40 total cost

## 🚨 Important Notes

- **cuDNN Fix**: All scripts include the necessary cuDNN library fix for PyTorch 2.8.0
- **First Run**: Initial setup downloads ~5GB of models (cached thereafter)
- **HuggingFace**: Required for speaker diarization features

## 🔗 Links

- [WhisperX GitHub](https://github.com/m-bain/whisperX)
- [Modal Documentation](https://modal.com/docs)
- [PyAnnote Audio](https://github.com/pyannote/pyannote-audio)

## 📝 License

WhisperX is licensed under BSD-2-Clause. See the main repository for details.

---

**Need help?** Check the detailed documentation files above or visit [Modal's support](https://modal.com/docs).
