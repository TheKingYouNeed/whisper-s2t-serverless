"""
WhisperS2T HTTP API Server for RunPod
FastAPI server with multipart file upload support for RapidAPI integration
"""

import os
import tempfile
import base64
import requests as http_requests
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

import whisper_s2t

app = FastAPI(title="WhisperS2T Transcription API")

# Global model instance
MODEL = None
READY = False

def load_model():
    """Load WhisperS2T model with CTranslate2 backend"""
    global MODEL, READY
    if MODEL is None:
        model_size = os.getenv("WHISPER_MODEL", "large-v3")
        backend = os.getenv("WHISPER_BACKEND", "CTranslate2")
        
        print(f"Loading WhisperS2T model: {model_size} with backend: {backend}")
        MODEL = whisper_s2t.load_model(
            model_identifier=model_size,
            backend=backend,
        )
        READY = True
        print("Model loaded successfully!")
    return MODEL


@app.get("/")
def root():
    """Root endpoint"""
    return {"service": "WhisperS2T Transcription API", "status": "ready" if READY else "loading"}


@app.get("/ping")
@app.get("/health")
def health_check():
    """Health check endpoint for load balancer - always returns 200"""
    return {"status": "ready" if READY else "loading", "model_loaded": READY}


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: Optional[str] = Form(default=None),
    task: str = Form(default="transcribe"),
    batch_size: int = Form(default=24),
):
    """
    Transcribe audio file using WhisperS2T
    
    Supports multipart/form-data file upload for RapidAPI integration
    
    Args:
        file: Audio file (mp3, wav, m4a, flac, etc.)
        language: Language code (e.g., 'en', 'fr') or None for auto-detect
        task: 'transcribe' or 'translate'
        batch_size: Batch size for VAD processing (default 24)
    
    Returns:
        JSON with transcription text and segments
    """
    if not READY or MODEL is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    
    # Save uploaded file to temp location
    suffix = os.path.splitext(file.filename or "audio.mp3")[1] or ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Transcribe with WhisperS2T VAD pipeline
        out = MODEL.transcribe_with_vad(
            [tmp_path],
            lang_codes=[language] if language else [None],
            tasks=[task],
            initial_prompts=[None],
            batch_size=batch_size,
        )
        
        # Process results
        segments = out[0] if out else []
        
        result_segments = []
        full_text_parts = []
        
        for seg in segments:
            seg_data = {
                "start": seg.get("start_time", 0),
                "end": seg.get("end_time", 0),
                "text": seg.get("text", "").strip(),
            }
            result_segments.append(seg_data)
            full_text_parts.append(seg.get("text", "").strip())
        
        return {
            "text": " ".join(full_text_parts),
            "segments": result_segments,
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup temp file
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


@app.post("/transcribe_url")
async def transcribe_url(
    audio_url: str = Form(...),
    language: Optional[str] = Form(default=None),
    task: str = Form(default="transcribe"),
    batch_size: int = Form(default=24),
):
    """
    Transcribe audio from URL
    Alternative endpoint for URL-based input
    """
    if not READY or MODEL is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    
    # Download audio
    try:
        response = http_requests.get(audio_url, timeout=300)
        response.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download audio: {e}")
    
    suffix = ".mp3"
    if "wav" in audio_url.lower():
        suffix = ".wav"
    elif "m4a" in audio_url.lower():
        suffix = ".m4a"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name
    
    try:
        out = MODEL.transcribe_with_vad(
            [tmp_path],
            lang_codes=[language] if language else [None],
            tasks=[task],
            initial_prompts=[None],
            batch_size=batch_size,
        )
        
        segments = out[0] if out else []
        full_text_parts = [seg.get("text", "").strip() for seg in segments]
        
        return {
            "text": " ".join(full_text_parts),
        }
        
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


@app.post("/transcribe_base64")
async def transcribe_base64(
    audio_base64: str = Form(...),
    language: Optional[str] = Form(default=None),
    task: str = Form(default="transcribe"),
    batch_size: int = Form(default=24),
):
    """
    Transcribe base64-encoded audio
    Alternative endpoint for base64 input
    """
    if not READY or MODEL is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    
    try:
        audio_data = base64.b64decode(audio_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 audio data")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(audio_data)
        tmp_path = tmp.name
    
    try:
        out = MODEL.transcribe_with_vad(
            [tmp_path],
            lang_codes=[language] if language else [None],
            tasks=[task],
            initial_prompts=[None],
            batch_size=batch_size,
        )
        
        segments = out[0] if out else []
        full_text_parts = [seg.get("text", "").strip() for seg in segments]
        
        return {
            "text": " ".join(full_text_parts),
        }
        
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


# Load model in background thread so server starts immediately
import threading

def _load_model_background():
    print("Loading WhisperS2T model in background...")
    load_model()
    print("Background model loading complete!")

print("Starting FastAPI server...")
threading.Thread(target=_load_model_background, daemon=True).start()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

