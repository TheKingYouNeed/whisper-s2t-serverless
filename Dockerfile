# WhisperS2T Load Balancing Serverless Dockerfile for RunPod
# Based on official WhisperS2T Dockerfile with FastAPI for multipart uploads

ARG BASE_IMAGE=nvidia/cuda
ARG BASE_TAG=12.1.0-devel-ubuntu22.04

FROM ${BASE_IMAGE}:${BASE_TAG}
ARG WHISPER_S2T_VER=main
ARG SKIP_TENSORRT_LLM=1

WORKDIR /app
ENTRYPOINT []
SHELL ["/bin/bash", "-c"]

# Install system dependencies and WhisperS2T
RUN apt update && apt-get install -y python3.10 python3-pip libsndfile1 ffmpeg git && \
    pip3 install --no-cache-dir git+https://github.com/shashikg/WhisperS2T.git@${WHISPER_S2T_VER} && \
    CUDNN_PATH=$(python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))') && \
    echo 'export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:'${CUDNN_PATH} >> ~/.bashrc

# Install FastAPI and dependencies for multipart upload support
RUN pip3 install --no-cache-dir \
    fastapi>=0.100.0 \
    uvicorn[standard]>=0.23.0 \
    python-multipart>=0.0.6

# Cleanup
RUN apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy application code
COPY src/app.py /app/app.py

# Environment variables - cost-effective settings
ENV WHISPER_MODEL=large-v3
ENV WHISPER_BACKEND=CTranslate2
ENV PORT=8000

# Pre-download model for faster cold starts (CTranslate2 backend - no TensorRT)
RUN python3 -c "import whisper_s2t; whisper_s2t.load_model(model_identifier='large-v3', backend='CTranslate2')" || echo "Model download will happen on first request"

# Expose port for Load Balancing
EXPOSE 8000

# Start FastAPI server with LD_LIBRARY_PATH set
CMD ["bash", "-c", "source ~/.bashrc && python3 -u /app/app.py"]
