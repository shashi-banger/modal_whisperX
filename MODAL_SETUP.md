# Running WhisperX on Modal

This guide shows how to run WhisperX on Modal's serverless GPU infrastructure.

## Prerequisites

1. **Install Modal**
```bash
pip install modal
```

2. **Setup Modal account**
```bash
modal setup
```
This will open a browser to authenticate with Modal.

3. **Setup HuggingFace token (for diarization only)**

If you want to use speaker diarization, you need a HuggingFace token:

a. Create a token at https://huggingface.co/settings/tokens
b. Accept the user agreements for:
   - https://huggingface.co/pyannote/segmentation-3.0
   - https://huggingface.co/pyannote/speaker-diarization-3.1

c. Create a Modal secret:
```bash
modal secret create huggingface-secret HF_TOKEN=your_token_here
```

## Usage

### Basic transcription (no diarization)

```bash
modal run modal_whisperx.py --audio-url "https://example.com/audio.mp3"
```

### With word-level alignment (default)

```bash
modal run modal_whisperx.py \
  --audio-url "https://example.com/audio.mp3" \
  --align
```

### With speaker diarization

```bash
modal run modal_whisperx.py \
  --audio-url "https://example.com/audio.mp3" \
  --diarize \
  --min-speakers 2 \
  --max-speakers 4
```

### Specify language (skip auto-detection)

```bash
modal run modal_whisperx.py \
  --audio-url "https://example.com/audio.mp3" \
  --language "en"
```

### Save output to file

```bash
modal run modal_whisperx.py \
  --audio-url "https://example.com/audio.mp3" \
  --diarize \
  --output-file "transcript.json"
```

### Process local file

First upload to Modal Volume, then process:

```bash
# See modal_whisperx_local.py for local file processing
modal run modal_whisperx_local.py --audio-file "./my_audio.mp3"
```

## Cost Considerations

- **GPU**: A100 costs ~$1.10/hour, H100 costs ~$4/hour
- **Storage**: Model cache persists across runs (free up to 50GB)
- **Scaledown**: Container stays warm for 10 minutes after last use

For batch processing, Modal parallelizes across containers automatically.

## Performance

On an A100:
- **ASR only**: ~70x realtime (1 hour audio in ~50 seconds)
- **With alignment**: ~60x realtime
- **With diarization**: ~40x realtime

## Customization

Edit `modal_whisperx.py` to customize:

- **Model size**: Change `"large-v2"` to `"medium"`, `"small"`, or `"large-v3"`
- **GPU type**: Change `GPU_CONFIG = modal.gpu.A100()` to `.H100()`, `.T4()`, `.L4()`, etc.
- **Batch size**: Increase for faster processing (if you have GPU memory)
- **Compute type**: Use `"int8"` for lower memory usage

## Troubleshooting

**Error: "Unable to load libcudnn_cnn.so"**
- This is a cuDNN library mismatch issue
- The scripts now include `nvidia-cudnn-cu12==9.8.0.87` and proper `LD_LIBRARY_PATH` configuration
- If you still see this error, the image needs to be rebuilt: `modal app stop whisperx-transcription` then retry
- See `MODAL_CUDNN_FIX.md` for detailed troubleshooting

**Error: "No module named whisperx"**
- The image build failed. Check Modal logs with `modal app logs`

**Error: "HF_TOKEN not found"**
- You need to set up the HuggingFace secret (see Prerequisites)
- Or run without `--diarize` flag

**Error: "CUDA out of memory"**
- Reduce `--batch-size` (default is 16)
- Use smaller model (`medium` or `small`)
- Use `compute_type="int8"` in the code

**Slow performance**
- First run downloads models (~5GB), subsequent runs use cache
- Make sure GPU is actually being used (check logs)

## Example Output

```
Starting WhisperX transcription...
Audio URL: https://example.com/audio.mp3
Language: auto-detect
Alignment: True
Diarization: True

Loading audio...
Transcribing...
Detected language: en
Performing alignment...
Performing speaker diarization...
Transcription complete!

================================================================================
TRANSCRIPTION RESULTS
================================================================================
Language: en
Segments: 42

[0.00s -> 3.50s] Speaker SPEAKER_00: Hello and welcome to this podcast.
[4.20s -> 8.10s] Speaker SPEAKER_01: Thanks for having me on the show.
[8.50s -> 12.30s] Speaker SPEAKER_00: Let's start with your background.
...

Full results saved to: transcript.json
```

## Advanced: Deploy as Web Service

To deploy as an always-on web service:

```python
@app.function(
    gpu=GPU_CONFIG,
    volumes={CACHE_DIR: cache_vol},
    keep_warm=1,  # Keep 1 container always warm
)
@modal.web_endpoint(method="POST")
def transcribe_web(item: dict):
    model = WhisperXModel()
    return model.transcribe.remote(
        audio_url=item["audio_url"],
        diarize=item.get("diarize", False),
    )
```

Then call via HTTP:
```bash
curl -X POST https://your-app.modal.run/transcribe_web \
  -H "Content-Type: application/json" \
  -d '{"audio_url": "https://example.com/audio.mp3", "diarize": true}'
```
