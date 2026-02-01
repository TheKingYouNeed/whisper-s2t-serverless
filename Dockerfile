# WhisperS2T HTTP Load Balancer Dockerfile for RunPod
# FastAPI server with multipart upload support for RapidAPI

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app
SHELL ["/bin/bash", "-c"]

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install WhisperS2T and FastAPI
RUN pip3 install --no-cache-dir \
    whisper-s2t \
    fastapi>=0.100.0 \
    uvicorn[standard]>=0.23.0 \
    python-multipart>=0.0.6 \
    requests

# Copy FastAPI application
COPY src/app.py /app/app.py

# Environment variables
ENV WHISPER_MODEL=large-v3
ENV WHISPER_BACKEND=CTranslate2
ENV PORT=8000

# Pre-download CTranslate2 model files (no CUDA needed for download)
# Using huggingface_hub to download without initializing GPU
RUN pip3 install --no-cache-dir huggingface_hub && \
    python3 -c "\
from huggingface_hub import snapshot_download; \
import os; \
models = ['tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'large-v3']; \
for i, m in enumerate(models, 1): \
    print(f'{i}/{len(models)} Downloading {m}...'); \
    repo = f'Systran/faster-whisper-{m}'; \
    snapshot_download(repo_id=repo, local_dir=f'/root/.cache/huggingface/hub/models--Systran--faster-whisper-{m}/snapshots/main'); \
print('All models downloaded!'); \
"

# Expose HTTP port
EXPOSE 8000

# Health check for load balancer
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI server
CMD ["python3", "-u", "/app/app.py"]
