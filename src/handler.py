"""
WhisperS2T Serverless Handler for RunPod
Uses CTranslate2 backend for fast, efficient transcription
"""

import os
import tempfile
import base64
import requests
import runpod
import whisper_s2t

# Global model instance
MODEL = None

def load_model():
    """Load WhisperS2T model with CTranslate2 backend"""
    global MODEL
    if MODEL is None:
        model_id = os.getenv("WHISPER_MODEL", "large-v3")
        backend = os.getenv("WHISPER_BACKEND", "CTranslate2")
        compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
        
        print(f"Loading WhisperS2T model: {model_id} with backend: {backend}")
        MODEL = whisper_s2t.load_model(
            model_identifier=model_id,
            backend=backend,
            compute_type=compute_type
        )
        print("Model loaded successfully!")
    return MODEL

def download_file(url: str, suffix: str = ".mp3") -> str:
    """Download file from URL to temp location"""
    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        for chunk in response.iter_content(chunk_size=8192):
            tmp.write(chunk)
        return tmp.name

def handler(job):
    """
    RunPod serverless handler for WhisperS2T transcription
    
    Input job format:
    {
        "input": {
            "audio_url": "https://example.com/audio.mp3",  # URL to audio file
            OR
            "audio_base64": "base64_encoded_audio_data",   # Base64 encoded audio
            
            "language": "en",              # Optional: language code
            "task": "transcribe",          # Optional: "transcribe" or "translate"
            "batch_size": 16,              # Optional: batch size for processing
            "return_timestamps": false     # Optional: include word timestamps
        }
    }
    """
    job_input = job.get("input", {})
    
    # Get audio source
    audio_url = job_input.get("audio_url")
    audio_base64 = job_input.get("audio_base64")
    
    if not audio_url and not audio_base64:
        return {"error": "Must provide either 'audio_url' or 'audio_base64'"}
    
    # Get optional parameters
    language = job_input.get("language")
    task = job_input.get("task", "transcribe")
    batch_size = job_input.get("batch_size", 16)
    return_timestamps = job_input.get("return_timestamps", False)
    
    # Load model
    model = load_model()
    
    # Get audio file
    tmp_path = None
    try:
        if audio_url:
            # Extract file extension from URL
            ext = os.path.splitext(audio_url.split("?")[0])[1] or ".mp3"
            tmp_path = download_file(audio_url, suffix=ext)
        else:
            # Decode base64 audio
            audio_data = base64.b64decode(audio_base64)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp.write(audio_data)
                tmp_path = tmp.name
        
        # Transcribe with VAD (Voice Activity Detection)
        result = model.transcribe_with_vad(
            [tmp_path],
            lang_codes=[language] if language else [None],
            tasks=[task],
            initial_prompts=[None],
            batch_size=batch_size,
        )
        
        # Process results
        segments = result[0] if result else []
        
        if return_timestamps:
            # Return segments with timestamps
            output_segments = []
            for seg in segments:
                output_segments.append({
                    "text": seg.get("text", "").strip(),
                    "start": seg.get("start_time", 0),
                    "end": seg.get("end_time", 0)
                })
            full_text = " ".join(seg["text"] for seg in output_segments)
            return {
                "text": full_text,
                "segments": output_segments
            }
        else:
            # Return just the text
            full_text = " ".join(seg.get("text", "").strip() for seg in segments)
            return {"text": full_text}
            
    except Exception as e:
        return {"error": str(e)}
    
    finally:
        # Cleanup temp file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

# Initialize model at cold start
print("Initializing WhisperS2T model...")
load_model()

# Start RunPod serverless handler
runpod.serverless.start({"handler": handler})
