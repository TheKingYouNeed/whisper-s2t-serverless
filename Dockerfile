# WhisperS2T Load Balancing Serverless Dockerfile for RunPod
# Simplified version for faster builds

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ARG WHISPER_S2T_VER=main

WORKDIR /app
SHELL ["/bin/bash", "-c"]

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install WhisperS2T (CTranslate2 backend - no TensorRT for simpler build)
RUN pip3 install --no-cache-dir git+https://github.com/shashikg/WhisperS2T.git@${WHISPER_S2T_VER}

# Install FastAPI and dependencies for multipart upload support
RUN pip3 install --no-cache-dir \
    fastapi>=0.100.0 \
    uvicorn[standard]>=0.23.0 \
    python-multipart>=0.0.6

# Copy application code
COPY src/app.py /app/app.py

# Environment variables - cost-effective settings
ENV WHISPER_MODEL=large-v3
ENV WHISPER_BACKEND=CTranslate2
ENV PORT=8000

# Pre-download model for faster cold starts
RUN python3 -c "import whisper_s2t; model = whisper_s2t.load_model(model_identifier='large-v3', backend='CTranslate2'); print('Model loaded successfully')" || echo "Model will download on first request"

# Expose port for Load Balancing
EXPOSE 8000

# Start FastAPI server
CMD ["python3", "-u", "/app/app.py"]
