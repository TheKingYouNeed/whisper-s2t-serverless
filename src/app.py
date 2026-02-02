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

# Model cache - supports multiple models
MODELS = {}
DEFAULT_MODEL = os.getenv("WHISPER_MODEL", "tiny")
BACKEND = os.getenv("WHISPER_BACKEND", "CTranslate2")

# Valid model names - MUST match models pre-downloaded in Dockerfile
# Only 4 models: tiny (fast), tiny.en (fast English), small (balanced), large-v3 (best)
VALID_MODELS = [
    "tiny", "tiny.en", 
    "small",
    "large-v3"
]

def get_model(model_name: str = None):
    """Get or load a WhisperS2T model"""
    global MODELS
    
    model_name = model_name or DEFAULT_MODEL
    
    # Validate model name
    if model_name not in VALID_MODELS:
        raise ValueError(f"Invalid model: {model_name}. Valid models: {VALID_MODELS}")
    
    # Load model if not cached
    if model_name not in MODELS:
        print(f"Loading WhisperS2T model: {model_name} with backend: {BACKEND}")
        MODELS[model_name] = whisper_s2t.load_model(
            model_identifier=model_name,
            backend=BACKEND,
        )
        print(f"Model {model_name} loaded successfully!")
    
    return MODELS[model_name]


@app.get("/")
def root():
    """Root endpoint"""
    return {
        "service": "WhisperS2T Transcription API", 
        "models_loaded": list(MODELS.keys()),
        "available_models": VALID_MODELS,
        "default_model": DEFAULT_MODEL
    }


@app.get("/ping")
@app.get("/health")
def health_check():
    """Health check endpoint for load balancer - always returns 200"""
    return {"status": "ready", "models_loaded": list(MODELS.keys())}


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    model: Optional[str] = Form(default=None),
    whisper_model: Optional[str] = Form(default=None),
    language: Optional[str] = Form(default=None),
    task: str = Form(default="transcribe"),
    output_format: str = Form(default="json"),
    word_timestamps: bool = Form(default=False),
    initial_prompt: Optional[str] = Form(default=None),
    batch_size: int = Form(default=24),
):
    """
    Transcribe audio file using WhisperS2T
    
    Supports multipart/form-data file upload for RapidAPI integration
    
    Args:
        file: Audio file (mp3, wav, m4a, flac, etc.)
        model: Model to use (tiny, tiny.en, base, base.en, small, small.en, medium, large-v3)
        language: Language code (e.g., 'en', 'fr', 'de', 'es', 'zh') - default 'en'
        task: 'transcribe' or 'translate' (translate to English)
        output_format: 'json', 'text', 'srt', or 'vtt'
        word_timestamps: Include word-level timestamps (slower)
        initial_prompt: Context/vocabulary hints for better accuracy
        batch_size: Batch size for VAD processing (default 24)
    
    Returns:
        Transcription in requested format
    """
    # Support both 'model' and 'whisper_model' parameters (RapidAPI compatibility)
    selected_model = whisper_model or model
    
    try:
        loaded_model = get_model(selected_model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Save uploaded file to temp location
    suffix = os.path.splitext(file.filename or "audio.mp3")[1] or ".mp3"
    content = await file.read()
    
    # Reject files > 100MB to prevent long processing
    if len(content) > 100 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Audio file too large (max 100MB)")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Transcribe with WhisperS2T VAD pipeline
        out = loaded_model.transcribe_with_vad(
            [tmp_path],
            lang_codes=[language] if language else ['en'],
            tasks=[task],
            initial_prompts=[initial_prompt],
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
            if word_timestamps and "word_timestamps" in seg:
                seg_data["words"] = seg["word_timestamps"]
            result_segments.append(seg_data)
            full_text_parts.append(seg.get("text", "").strip())
        
        full_text = " ".join(full_text_parts)
        
        # Return in requested format
        if output_format == "text":
            return {"text": full_text}
        elif output_format == "srt":
            return {"srt": _to_srt(result_segments)}
        elif output_format == "vtt":
            return {"vtt": _to_vtt(result_segments)}
        else:  # json
            return {
                "text": full_text,
                "segments": result_segments,
                "model": selected_model or DEFAULT_MODEL,
                "language": language or "en",
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


def _to_srt(segments):
    """Convert segments to SRT format"""
    lines = []
    for i, seg in enumerate(segments, 1):
        start = _format_timestamp_srt(seg["start"])
        end = _format_timestamp_srt(seg["end"])
        lines.append(f"{i}\n{start} --> {end}\n{seg['text']}\n")
    return "\n".join(lines)


def _to_vtt(segments):
    """Convert segments to VTT format"""
    lines = ["WEBVTT\n"]
    for seg in segments:
        start = _format_timestamp_vtt(seg["start"])
        end = _format_timestamp_vtt(seg["end"])
        lines.append(f"{start} --> {end}\n{seg['text']}\n")
    return "\n".join(lines)


def _format_timestamp_srt(seconds):
    """Format seconds to SRT timestamp (HH:MM:SS,mmm)"""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{ms:03d}"


def _format_timestamp_vtt(seconds):
    """Format seconds to VTT timestamp (HH:MM:SS.mmm)"""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}.{ms:03d}"


@app.post("/transcribe_url")
async def transcribe_url(
    audio_url: str = Form(...),
    model: Optional[str] = Form(default=None),
    whisper_model: Optional[str] = Form(default=None),
    language: Optional[str] = Form(default=None),
    task: str = Form(default="transcribe"),
    output_format: str = Form(default="json"),
    word_timestamps: bool = Form(default=False),
    initial_prompt: Optional[str] = Form(default=None),
    batch_size: int = Form(default=24),
):
    """
    Transcribe audio from URL
    
    Args:
        audio_url: URL to audio file
        model: Model to use (tiny, small, medium, large-v3, etc.)
        language: Language code (e.g., 'en', 'fr')
        task: 'transcribe' or 'translate'
        output_format: 'json', 'text', 'srt', or 'vtt'
        word_timestamps: Include word-level timestamps
        initial_prompt: Context hints for accuracy
        batch_size: VAD batch size
    """
    # Support both 'model' and 'whisper_model' parameters (RapidAPI compatibility)
    selected_model = whisper_model or model
    
    try:
        loaded_model = get_model(selected_model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Download audio with reasonable timeout
    tmp_path = None
    try:
        response = http_requests.get(audio_url, timeout=60, stream=True)
        response.raise_for_status()
    except http_requests.Timeout:
        raise HTTPException(status_code=408, detail="Audio download timed out (60s limit)")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download audio: {e}")
    
    # Check content length - reject files > 100MB to prevent long processing
    content_length = response.headers.get('content-length')
    if content_length and int(content_length) > 100 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Audio file too large (max 100MB)")
    
    suffix = ".mp3"
    if "wav" in audio_url.lower():
        suffix = ".wav"
    elif "m4a" in audio_url.lower():
        suffix = ".m4a"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        for chunk in response.iter_content(chunk_size=8192):
            tmp.write(chunk)
        tmp_path = tmp.name
    
    try:
        out = loaded_model.transcribe_with_vad(
            [tmp_path],
            lang_codes=[language] if language else ['en'],
            tasks=[task],
            initial_prompts=[initial_prompt],
            batch_size=batch_size,
        )
        
        segments = out[0] if out else []
        result_segments = []
        full_text_parts = []
        
        for seg in segments:
            seg_data = {
                "start": seg.get("start_time", 0),
                "end": seg.get("end_time", 0),
                "text": seg.get("text", "").strip(),
            }
            if word_timestamps and "word_timestamps" in seg:
                seg_data["words"] = seg["word_timestamps"]
            result_segments.append(seg_data)
            full_text_parts.append(seg.get("text", "").strip())
        
        full_text = " ".join(full_text_parts)
        
        if output_format == "text":
            return {"text": full_text}
        elif output_format == "srt":
            return {"srt": _to_srt(result_segments)}
        elif output_format == "vtt":
            return {"vtt": _to_vtt(result_segments)}
        else:
            return {
                "text": full_text,
                "segments": result_segments,
                "model": selected_model or DEFAULT_MODEL,
                "language": language or "en",
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


@app.post("/transcribe_base64")
async def transcribe_base64(
    audio_base64: str = Form(...),
    model: Optional[str] = Form(default=None),
    language: Optional[str] = Form(default=None),
    task: str = Form(default="transcribe"),
    output_format: str = Form(default="json"),
    word_timestamps: bool = Form(default=False),
    initial_prompt: Optional[str] = Form(default=None),
    batch_size: int = Form(default=24),
):
    """
    Transcribe base64-encoded audio
    
    Args:
        audio_base64: Base64-encoded audio data
        model: Model to use (tiny, small, medium, large-v3, etc.)
        language: Language code (e.g., 'en', 'fr')
        task: 'transcribe' or 'translate'
        output_format: 'json', 'text', 'srt', or 'vtt'
        word_timestamps: Include word-level timestamps
        initial_prompt: Context hints for accuracy
        batch_size: VAD batch size
    """
    try:
        whisper_model = get_model(model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    tmp_path = None
    try:
        audio_data = base64.b64decode(audio_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 audio data")
    
    # Reject files > 100MB to prevent long processing
    if len(audio_data) > 100 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Audio file too large (max 100MB)")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(audio_data)
        tmp_path = tmp.name
    
    try:
        out = whisper_model.transcribe_with_vad(
            [tmp_path],
            lang_codes=[language] if language else ['en'],
            tasks=[task],
            initial_prompts=[initial_prompt],
            batch_size=batch_size,
        )
        
        segments = out[0] if out else []
        result_segments = []
        full_text_parts = []
        
        for seg in segments:
            seg_data = {
                "start": seg.get("start_time", 0),
                "end": seg.get("end_time", 0),
                "text": seg.get("text", "").strip(),
            }
            if word_timestamps and "word_timestamps" in seg:
                seg_data["words"] = seg["word_timestamps"]
            result_segments.append(seg_data)
            full_text_parts.append(seg.get("text", "").strip())
        
        full_text = " ".join(full_text_parts)
        
        if output_format == "text":
            return {"text": full_text}
        elif output_format == "srt":
            return {"srt": _to_srt(result_segments)}
        elif output_format == "vtt":
            return {"vtt": _to_vtt(result_segments)}
        else:
            return {
                "text": full_text,
                "segments": result_segments,
                "model": model or DEFAULT_MODEL,
                "language": language or "en",
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


# Preload ALL models at startup for instant model switching (no reload overhead)
import time

def preload_all_models():
    """Load all models into VRAM at startup for zero-overhead model switching"""
    print(f"Preloading ALL {len(VALID_MODELS)} models into memory...")
    start_time = time.time()
    for i, model in enumerate(VALID_MODELS, 1):
        print(f"  [{i}/{len(VALID_MODELS)}] Loading {model}...")
        try:
            get_model(model)
        except Exception as e:
            print(f"  Warning: Failed to load {model}: {e}")
    elapsed = time.time() - start_time
    print(f"All models preloaded in {elapsed:.1f}s. Ready for instant inference!")

# Preload all models at startup
preload_all_models()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
