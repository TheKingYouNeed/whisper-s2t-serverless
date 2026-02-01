# WhisperS2T RunPod Serverless Dockerfile
# Uses proper RunPod handler for queue-based serverless

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app
SHELL ["/bin/bash", "-c"]

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install WhisperS2T and RunPod SDK
RUN pip3 install --no-cache-dir \
    whisper-s2t \
    runpod \
    requests

# Copy handler code
COPY src/handler.py /app/handler.py

# Environment variables
ENV WHISPER_MODEL=large-v3
ENV WHISPER_BACKEND=CTranslate2

# Pre-download model for faster cold starts
RUN python3 -c "import whisper_s2t; model = whisper_s2t.load_model(model_identifier='large-v3', backend='CTranslate2'); print('Model loaded successfully')" || echo "Model will download on first request"

# Start RunPod serverless handler
CMD ["python3", "-u", "/app/handler.py"]
