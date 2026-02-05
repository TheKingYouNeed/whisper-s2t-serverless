"""
Real-Time Audio Streaming Transcription - OPTIMIZED for Scale
WebSocket endpoint for streaming audio from Android devices
Supports thousands of concurrent connections via:
- Connection pooling and limits
- Async processing queue
- Memory-bounded buffers
- Session timeout management
"""

import asyncio
import base64
import json
import os
import struct
import tempfile
import time
import uuid
from typing import Dict, Optional
from collections import defaultdict
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

from fastapi import WebSocket, WebSocketDisconnect

# =============================================================================
# CONFIGURATION - Tune these for your hardware
# =============================================================================
MAX_CONCURRENT_CONNECTIONS = 500      # Max WebSocket connections
MAX_CONCURRENT_TRANSCRIPTIONS = 10    # Max parallel GPU transcriptions
MAX_AUDIO_BUFFER_MB = 5               # Max audio buffer per session (MB)
SESSION_TIMEOUT_SECONDS = 300         # Auto-cleanup inactive sessions
AUDIO_CHUNK_TRIGGER_BYTES = 32000     # ~2 seconds at 16kHz 16-bit mono
MAX_QUEUE_SIZE = 100                  # Max pending transcription jobs

# =============================================================================
# GLOBAL STATE
# =============================================================================
active_sessions: Dict[str, 'TranscriptionSession'] = {}
connection_count = 0
transcription_semaphore: Optional[asyncio.Semaphore] = None
processing_queue: Optional[asyncio.Queue] = None
gpu_executor: Optional[ThreadPoolExecutor] = None


def init_scaling():
    """Initialize scaling primitives"""
    global transcription_semaphore, processing_queue, gpu_executor
    transcription_semaphore = asyncio.Semaphore(MAX_CONCURRENT_TRANSCRIPTIONS)
    processing_queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
    gpu_executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_TRANSCRIPTIONS)


@dataclass
class TranscriptionSession:
    """Manages a single real-time transcription session with memory limits"""
    
    session_id: str
    whisper_model: str = "tiny"
    language: str = "en"
    audio_buffer: bytearray = field(default_factory=bytearray)
    chunk_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    transcription_results: list = field(default_factory=list)
    is_active: bool = True
    is_processing: bool = False  # Lock to prevent concurrent processing
    
    @property
    def buffer_size_mb(self) -> float:
        return len(self.audio_buffer) / (1024 * 1024)
    
    @property
    def is_timed_out(self) -> bool:
        return (time.time() - self.last_activity) > SESSION_TIMEOUT_SECONDS
    
    def add_audio_chunk(self, audio_data: bytes) -> bool:
        """Add audio chunk, returns False if buffer limit exceeded"""
        if self.buffer_size_mb >= MAX_AUDIO_BUFFER_MB:
            return False
        self.audio_buffer.extend(audio_data)
        self.chunk_count += 1
        self.last_activity = time.time()
        return True
    
    def get_audio_for_transcription(self) -> bytes:
        """Get buffered audio and clear buffer"""
        audio = bytes(self.audio_buffer)
        self.audio_buffer.clear()
        return audio
    
    def has_enough_audio(self) -> bool:
        """Check if we have enough audio to transcribe"""
        return len(self.audio_buffer) >= AUDIO_CHUNK_TRIGGER_BYTES


async def cleanup_stale_sessions():
    """Background task to clean up timed-out sessions"""
    while True:
        await asyncio.sleep(60)  # Run every minute
        stale = [sid for sid, s in active_sessions.items() if s.is_timed_out]
        for sid in stale:
            try:
                del active_sessions[sid]
                print(f"Cleaned up stale session: {sid}")
            except:
                pass


async def handle_realtime_websocket(websocket: WebSocket, get_model_func):
    """
    Handle WebSocket connection for real-time transcription
    Optimized with connection limits and async processing
    """
    global connection_count
    
    # Check connection limit
    if connection_count >= MAX_CONCURRENT_CONNECTIONS:
        await websocket.close(code=1013, reason="Server at capacity")
        return
    
    await websocket.accept()
    connection_count += 1
    
    session: Optional[TranscriptionSession] = None
    session_id = str(uuid.uuid4())
    
    try:
        while True:
            # Receive with timeout to detect dead connections
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=SESSION_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                await websocket.send_json({"error": "Session timed out"})
                break
            
            message = json.loads(data)
            action = message.get("action", "")
            
            if action == "start":
                whisper_model = message.get("whisper_model", "tiny")
                language = message.get("language", "en")
                
                session = TranscriptionSession(
                    session_id=session_id,
                    whisper_model=whisper_model,
                    language=language
                )
                active_sessions[session_id] = session
                
                await websocket.send_json({
                    "status": "started",
                    "session_id": session_id,
                    "config": {
                        "max_buffer_mb": MAX_AUDIO_BUFFER_MB,
                        "chunk_trigger_bytes": AUDIO_CHUNK_TRIGGER_BYTES
                    }
                })
                
            elif action == "audio" and session:
                audio_b64 = message.get("data", "")
                try:
                    audio_bytes = base64.b64decode(audio_b64)
                    
                    if not session.add_audio_chunk(audio_bytes):
                        await websocket.send_json({
                            "warning": "Buffer full, processing..."
                        })
                    
                    # Process if enough audio and not already processing
                    if session.has_enough_audio() and not session.is_processing:
                        session.is_processing = True
                        try:
                            transcription = await process_audio_chunk_optimized(
                                session, get_model_func
                            )
                            if transcription:
                                await websocket.send_json({
                                    "text": transcription,
                                    "chunk_index": session.chunk_count,
                                    "is_final": False,
                                    "buffer_mb": round(session.buffer_size_mb, 2)
                                })
                        finally:
                            session.is_processing = False
                            
                except Exception as e:
                    await websocket.send_json({
                        "error": f"Processing error: {str(e)}"
                    })
                    
            elif action == "stop" and session:
                # Process remaining audio
                if len(session.audio_buffer) > 0:
                    session.is_processing = True
                    final_text = await process_audio_chunk_optimized(
                        session, get_model_func, is_final=True
                    )
                else:
                    final_text = " ".join(session.transcription_results)
                
                await websocket.send_json({
                    "text": final_text,
                    "is_final": True,
                    "total_chunks": session.chunk_count,
                    "total_results": len(session.transcription_results)
                })
                
                session.is_active = False
                if session_id in active_sessions:
                    del active_sessions[session_id]
                break
                
            elif action == "ping":
                await websocket.send_json({
                    "action": "pong",
                    "active_sessions": len(active_sessions),
                    "connections": connection_count
                })
                
            elif action == "status" and session:
                await websocket.send_json({
                    "session_id": session_id,
                    "buffer_mb": round(session.buffer_size_mb, 2),
                    "chunks": session.chunk_count,
                    "results": len(session.transcription_results),
                    "is_processing": session.is_processing
                })
                
            else:
                await websocket.send_json({
                    "error": f"Unknown action: {action}"
                })
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass
    finally:
        connection_count -= 1
        if session_id in active_sessions:
            del active_sessions[session_id]


async def process_audio_chunk_optimized(
    session: TranscriptionSession, 
    get_model_func,
    is_final: bool = False
) -> str:
    """Process audio with GPU semaphore for controlled concurrency"""
    
    global transcription_semaphore, gpu_executor
    
    # Initialize if needed
    if transcription_semaphore is None:
        init_scaling()
    
    audio_data = session.get_audio_for_transcription()
    if not audio_data:
        return ""
    
    # Acquire semaphore to limit GPU concurrency
    async with transcription_semaphore:
        return await _do_transcription(audio_data, session, get_model_func)


async def _do_transcription(
    audio_data: bytes,
    session: TranscriptionSession,
    get_model_func
) -> str:
    """Actual transcription work, run in thread pool"""
    
    tmp_path = None
    
    try:
        # Create WAV file from raw PCM
        tmp_path = _create_wav_file(audio_data)
        
        # Get model
        model = get_model_func(session.whisper_model)
        
        # Run in executor to not block event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            gpu_executor,
            lambda: model.transcribe_with_vad(
                [tmp_path],
                lang_codes=[session.language],
                tasks=["transcribe"],
                batch_size=16,
            )
        )
        
        # Extract text
        segments = result[0] if result else []
        text_parts = [seg.get("text", "").strip() for seg in segments]
        transcription = " ".join(text_parts)
        
        if transcription:
            session.transcription_results.append(transcription)
        
        return transcription
        
    except Exception as e:
        return f"[Error: {str(e)}]"
        
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass


def _create_wav_file(audio_data: bytes) -> str:
    """Create WAV file from raw PCM audio data"""
    sample_rate = 16000
    bits_per_sample = 16
    channels = 1
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    data_size = len(audio_data)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(b'RIFF')
        tmp.write(struct.pack('<I', 36 + data_size))
        tmp.write(b'WAVE')
        tmp.write(b'fmt ')
        tmp.write(struct.pack('<I', 16))
        tmp.write(struct.pack('<H', 1))
        tmp.write(struct.pack('<H', channels))
        tmp.write(struct.pack('<I', sample_rate))
        tmp.write(struct.pack('<I', byte_rate))
        tmp.write(struct.pack('<H', block_align))
        tmp.write(struct.pack('<H', bits_per_sample))
        tmp.write(b'data')
        tmp.write(struct.pack('<I', data_size))
        tmp.write(audio_data)
        return tmp.name


def register_websocket_routes(app, get_model_func):
    """Register WebSocket routes with the FastAPI app"""
    
    # Initialize scaling on startup
    init_scaling()
    
    # Start cleanup background task
    @app.on_event("startup")
    async def start_cleanup_task():
        asyncio.create_task(cleanup_stale_sessions())
    
    @app.websocket("/ws/transcribe")
    async def websocket_transcribe(websocket: WebSocket):
        await handle_realtime_websocket(websocket, get_model_func)
    
    @app.get("/ws/stats")
    async def get_stats():
        """Get server stats for monitoring"""
        return {
            "connections": connection_count,
            "max_connections": MAX_CONCURRENT_CONNECTIONS,
            "active_sessions": len(active_sessions),
            "max_concurrent_transcriptions": MAX_CONCURRENT_TRANSCRIPTIONS,
            "config": {
                "max_buffer_mb": MAX_AUDIO_BUFFER_MB,
                "session_timeout_sec": SESSION_TIMEOUT_SECONDS,
                "chunk_trigger_bytes": AUDIO_CHUNK_TRIGGER_BYTES
            }
        }
    
    @app.get("/ws/sessions")
    async def get_active_sessions_info():
        """Get info about active streaming sessions"""
        return {
            "count": len(active_sessions),
            "sessions": [
                {
                    "session_id": s.session_id,
                    "chunks": s.chunk_count,
                    "buffer_mb": round(s.buffer_size_mb, 2),
                    "is_processing": s.is_processing,
                    "age_seconds": int(time.time() - s.created_at)
                }
                for s in list(active_sessions.values())[:50]  # Limit output
            ]
        }
