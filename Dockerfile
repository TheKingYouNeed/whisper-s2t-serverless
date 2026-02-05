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
    requests \
    websockets \
    aiofiles

# Copy FastAPI application and realtime streaming module
COPY src/app.py /app/app.py
COPY src/realtime_streaming.py /app/realtime_streaming.py

# Environment variables
ENV WHISPER_MODEL=large-v3
ENV WHISPER_BACKEND=CTranslate2
ENV PORT=8000

# Pre-download CTranslate2 model files (no CUDA needed for download)
COPY src/download_models.py /app/download_models.py
RUN pip3 install --no-cache-dir huggingface_hub && python3 /app/download_models.py

# Expose HTTP port
EXPOSE 8000

# Health check for load balancer
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI server
CMD ["python3", "-u", "/app/app.py"]
