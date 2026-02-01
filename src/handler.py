"""
WhisperS2T RunPod Serverless Handler
Supports both queue-based requests and HTTP proxy for RapidAPI
"""

import os
import tempfile
import base64
import requests
from typing import Optional

import runpod
import whisper_s2t

# Global model instance
MODEL = None

def load_model():
    """Load WhisperS2T model with CTranslate2 backend"""
    global MODEL
    if MODEL is None:
        model_size = os.getenv("WHISPER_MODEL", "large-v3")
        backend = os.getenv("WHISPER_BACKEND", "CTranslate2")
        
        print(f"Loading WhisperS2T model: {model_size} with backend: {backend}")
        MODEL = whisper_s2t.load_model(
            model_identifier=model_size,
            backend=backend,
        )
        print("Model loaded successfully!")
    return MODEL


def download_audio(audio_url: str) -> str:
    """Download audio from URL to temp file"""
    response = requests.get(audio_url, timeout=300)
    response.raise_for_status()
    
    suffix = ".mp3"
    if "wav" in audio_url.lower():
        suffix = ".wav"
    elif "m4a" in audio_url.lower():
        suffix = ".m4a"
    elif "flac" in audio_url.lower():
        suffix = ".flac"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(response.content)
        return tmp.name


def decode_base64_audio(audio_base64: str) -> str:
    """Decode base64 audio to temp file"""
    audio_data = base64.b64decode(audio_base64)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(audio_data)
        return tmp.name


def transcribe_audio(audio_path: str, language: Optional[str] = None, task: str = "transcribe", batch_size: int = 24):
    """Transcribe audio file using WhisperS2T"""
    model = load_model()
    
    out = model.transcribe_with_vad(
        [audio_path],
        lang_codes=[language] if language else [None],
        tasks=[task],
        initial_prompts=[None],
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
        result_segments.append(seg_data)
        full_text_parts.append(seg.get("text", "").strip())
    
    return {
        "text": " ".join(full_text_parts),
        "segments": result_segments,
    }


def handler(job):
    """
    RunPod Serverless Handler
    
    Accepts:
        - audio: URL to audio file
        - audio_base64: Base64 encoded audio (alternative to URL)
        - language: Optional language code
        - task: 'transcribe' or 'translate'
        - batch_size: Batch size for processing
    """
    job_input = job.get("input", {})
    
    audio_url = job_input.get("audio")
    audio_base64 = job_input.get("audio_base64")
    language = job_input.get("language")
    task = job_input.get("task", "transcribe")
    batch_size = job_input.get("batch_size", 24)
    
    if not audio_url and not audio_base64:
        return {"error": "Either 'audio' (URL) or 'audio_base64' is required"}
    
    audio_path = None
    try:
        # Get audio file
        if audio_url:
            print(f"Downloading audio from: {audio_url}")
            audio_path = download_audio(audio_url)
        else:
            print("Decoding base64 audio")
            audio_path = decode_base64_audio(audio_base64)
        
        # Transcribe
        print(f"Transcribing: {audio_path}")
        result = transcribe_audio(audio_path, language, task, batch_size)
        
        return result
        
    except Exception as e:
        return {"error": str(e)}
    
    finally:
        # Cleanup
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except OSError:
                pass


# Pre-load model at startup for faster cold starts
print("Initializing WhisperS2T model at startup...")
try:
    load_model()
    print("Model pre-loaded successfully!")
except Exception as e:
    print(f"Warning: Model pre-load failed: {e}")

# Start RunPod serverless handler
runpod.serverless.start({"handler": handler})
