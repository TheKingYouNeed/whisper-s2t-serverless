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

# Pre-download ALL models for instant loading at runtime
# This adds ~10GB to image but ensures no download delays
RUN python3 -c "\
import whisper_s2t; \
print('Downloading tiny...'); whisper_s2t.load_model('tiny', backend='CTranslate2'); \
print('Downloading tiny.en...'); whisper_s2t.load_model('tiny.en', backend='CTranslate2'); \
print('Downloading base...'); whisper_s2t.load_model('base', backend='CTranslate2'); \
print('Downloading base.en...'); whisper_s2t.load_model('base.en', backend='CTranslate2'); \
print('Downloading small...'); whisper_s2t.load_model('small', backend='CTranslate2'); \
print('Downloading small.en...'); whisper_s2t.load_model('small.en', backend='CTranslate2'); \
print('Downloading medium...'); whisper_s2t.load_model('medium', backend='CTranslate2'); \
print('Downloading medium.en...'); whisper_s2t.load_model('medium.en', backend='CTranslate2'); \
print('Downloading large-v2...'); whisper_s2t.load_model('large-v2', backend='CTranslate2'); \
print('Downloading large-v3...'); whisper_s2t.load_model('large-v3', backend='CTranslate2'); \
print('All models downloaded!'); \
"

# Expose HTTP port
EXPOSE 8000

# Health check for load balancer
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI server
CMD ["python3", "-u", "/app/app.py"]
