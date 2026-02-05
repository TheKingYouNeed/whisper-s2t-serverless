"""
Real-Time Audio Streaming Transcription
WebSocket endpoint for streaming audio from Android devices
"""

import asyncio
import base64
import json
import os
import tempfile
import time
import uuid
from typing import Dict, Optional
from collections import defaultdict

from fastapi import WebSocket, WebSocketDisconnect
import aiofiles

# Session storage for active transcription sessions
active_sessions: Dict[str, dict] = {}

# Audio chunk buffer per session
audio_buffers: Dict[str, bytearray] = defaultdict(bytearray)


class TranscriptionSession:
    """Manages a single real-time transcription session"""
    
    def __init__(self, session_id: str, whisper_model: str = "tiny", language: str = "en"):
        self.session_id = session_id
        self.whisper_model = whisper_model
        self.language = language
        self.audio_buffer = bytearray()
        self.chunk_count = 0
        self.created_at = time.time()
        self.last_activity = time.time()
        self.transcription_results = []
        self.is_active = True
        
    def add_audio_chunk(self, audio_data: bytes):
        """Add audio chunk to buffer"""
        self.audio_buffer.extend(audio_data)
        self.chunk_count += 1
        self.last_activity = time.time()
        
    def get_audio_for_transcription(self) -> bytes:
        """Get buffered audio and clear buffer"""
        audio = bytes(self.audio_buffer)
        self.audio_buffer.clear()
        return audio
    
    def has_enough_audio(self, min_bytes: int = 16000) -> bool:
        """Check if we have enough audio to transcribe (default ~1 second at 16kHz)"""
        return len(self.audio_buffer) >= min_bytes


async def handle_realtime_websocket(websocket: WebSocket, get_model_func):
    """
    Handle WebSocket connection for real-time transcription
    
    Protocol:
    1. Client sends: {"action": "start", "whisper_model": "tiny", "language": "en"}
    2. Client sends: {"action": "audio", "data": "<base64 audio chunk>"}
    3. Server sends: {"text": "partial transcription", "is_final": false}
    4. Client sends: {"action": "stop"}
    5. Server sends: {"text": "final transcription", "is_final": true}
    """
    await websocket.accept()
    
    session: Optional[TranscriptionSession] = None
    session_id = str(uuid.uuid4())
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            action = message.get("action", "")
            
            if action == "start":
                # Initialize new session
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
                    "message": "Send audio chunks with action='audio'"
                })
                
            elif action == "audio" and session:
                # Receive audio chunk
                audio_b64 = message.get("data", "")
                try:
                    audio_bytes = base64.b64decode(audio_b64)
                    session.add_audio_chunk(audio_bytes)
                    
                    # Process if we have enough audio (every ~2 seconds of audio)
                    if session.has_enough_audio(min_bytes=32000):
                        transcription = await process_audio_chunk(
                            session, get_model_func
                        )
                        if transcription:
                            await websocket.send_json({
                                "text": transcription,
                                "chunk_index": session.chunk_count,
                                "is_final": False
                            })
                except Exception as e:
                    await websocket.send_json({
                        "error": f"Audio processing error: {str(e)}"
                    })
                    
            elif action == "stop" and session:
                # Finalize session - process remaining audio
                if len(session.audio_buffer) > 0:
                    final_transcription = await process_audio_chunk(
                        session, get_model_func, is_final=True
                    )
                else:
                    final_transcription = " ".join(session.transcription_results)
                
                await websocket.send_json({
                    "text": final_transcription,
                    "is_final": True,
                    "total_chunks": session.chunk_count
                })
                
                # Cleanup
                session.is_active = False
                if session_id in active_sessions:
                    del active_sessions[session_id]
                break
                
            elif action == "ping":
                # Keepalive
                await websocket.send_json({"action": "pong"})
                
            else:
                await websocket.send_json({
                    "error": f"Unknown action: {action}"
                })
                
    except WebSocketDisconnect:
        # Client disconnected
        if session_id in active_sessions:
            del active_sessions[session_id]
    except Exception as e:
        await websocket.send_json({"error": str(e)})
        if session_id in active_sessions:
            del active_sessions[session_id]


async def process_audio_chunk(
    session: TranscriptionSession, 
    get_model_func,
    is_final: bool = False
) -> str:
    """Process accumulated audio and return transcription"""
    
    audio_data = session.get_audio_for_transcription()
    if not audio_data:
        return ""
    
    # Save to temp file for transcription
    suffix = ".wav"
    tmp_path = None
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            # Write WAV header for raw PCM data
            # Assuming 16kHz, 16-bit, mono
            import struct
            sample_rate = 16000
            bits_per_sample = 16
            channels = 1
            byte_rate = sample_rate * channels * bits_per_sample // 8
            block_align = channels * bits_per_sample // 8
            data_size = len(audio_data)
            
            # WAV header
            tmp.write(b'RIFF')
            tmp.write(struct.pack('<I', 36 + data_size))
            tmp.write(b'WAVE')
            tmp.write(b'fmt ')
            tmp.write(struct.pack('<I', 16))  # Subchunk1Size
            tmp.write(struct.pack('<H', 1))   # AudioFormat (PCM)
            tmp.write(struct.pack('<H', channels))
            tmp.write(struct.pack('<I', sample_rate))
            tmp.write(struct.pack('<I', byte_rate))
            tmp.write(struct.pack('<H', block_align))
            tmp.write(struct.pack('<H', bits_per_sample))
            tmp.write(b'data')
            tmp.write(struct.pack('<I', data_size))
            tmp.write(audio_data)
            tmp_path = tmp.name
        
        # Get model and transcribe
        model = get_model_func(session.whisper_model)
        
        # Run transcription (in executor to not block)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: model.transcribe_with_vad(
                [tmp_path],
                lang_codes=[session.language],
                tasks=["transcribe"],
                batch_size=16,  # Smaller batch for streaming
            )
        )
        
        # Extract text
        segments = result[0] if result else []
        text_parts = [seg.get("text", "").strip() for seg in segments]
        transcription = " ".join(text_parts)
        
        # Store result
        if transcription:
            session.transcription_results.append(transcription)
        
        return transcription
        
    except Exception as e:
        return f"[Error: {str(e)}]"
        
    finally:
        # Cleanup temp file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass


def register_websocket_routes(app, get_model_func):
    """Register WebSocket routes with the FastAPI app"""
    
    @app.websocket("/ws/transcribe")
    async def websocket_transcribe(websocket: WebSocket):
        await handle_realtime_websocket(websocket, get_model_func)
    
    @app.get("/ws/sessions")
    async def get_active_sessions():
        """Get info about active streaming sessions"""
        return {
            "active_sessions": len(active_sessions),
            "sessions": [
                {
                    "session_id": s.session_id,
                    "chunks": s.chunk_count,
                    "active_seconds": int(time.time() - s.created_at)
                }
                for s in active_sessions.values()
            ]
        }
