# WhisperS2T Serverless API for RapidAPI

Fast, cost-effective audio transcription API powered by WhisperS2T with CTranslate2 backend.

## Features

- 🚀 **Optimized Pipeline**: Uses WhisperS2T with CTranslate2 for fast inference
- 📤 **Multipart Upload**: Direct file upload support for RapidAPI integration
- 💰 **Serverless**: Scales to zero, pay only for what you use
- 🎯 **Cheap GPUs**: Prioritizes RTX 3090 and A5000 for cost savings

## API Endpoints

### POST /transcribe
Upload audio file directly (multipart/form-data)

**Parameters:**
- `file` (required): Audio file (mp3, wav, m4a, etc.)
- `language` (optional): Language code (e.g., 'en', 'fr') or auto-detect
- `task` (optional): 'transcribe' or 'translate' (default: transcribe)
- `batch_size` (optional): Batch size for processing (default: 24)

**Response:**
```json
{
  "text": "Full transcription text...",
  "segments": [
    {"start": 0.0, "end": 2.5, "text": "First segment..."},
    {"start": 2.5, "end": 5.0, "text": "Second segment..."}
  ]
}
```

### POST /transcribe_base64
Send base64-encoded audio

### GET /ping or /health
Health check endpoint

## Deployment

1. Push to GitHub (triggers automatic Docker build)
2. Wait for GitHub Actions to complete (~15-20 min)
3. Copy the GHCR image URL from Actions output
4. Create RunPod Load Balancing endpoint with the image

## RunPod Configuration

- **GPU Types**: AMPERE_24 (RTX 3090, A5000 - cheapest)
- **Workers Min**: 0 (serverless, scale to zero)
- **Workers Max**: 3
- **Idle Timeout**: 60 seconds
