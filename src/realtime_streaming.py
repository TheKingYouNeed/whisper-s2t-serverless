"""
Real-Time Audio Streaming Transcription - VERTICALLY SCALED
Optimized for MAXIMUM throughput on a single GPU instance

Key Optimizations:
1. Batched GPU inference - combine multiple sessions into one GPU call
2. Shared memory buffers - reduce memory copies
3. Optimized audio processing - skip unnecessary conversions
4. Connection pooling - handle 1000+ WebSockets
5. Async everything - never block the event loop
"""

import asyncio
import base64
import json
import os
import struct
import tempfile
import time
import uuid
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import threading

from fastapi import WebSocket, WebSocketDisconnect

# =============================================================================
# VERTICAL SCALING CONFIGURATION
# Tuned for 100x real-time GPU (RTX 3080 Ti / RTX 4080 class)
# =============================================================================

# Connection limits (per process)
MAX_CONCURRENT_CONNECTIONS = 5000     # WebSocket connections (CPU bound)
CONNECTION_TIMEOUT_SECONDS = 300      # 5 min idle timeout

# GPU Processing - AGGRESSIVE for 100x real-time GPU
GPU_BATCH_SIZE = 32                   # Process 32 transcriptions per GPU call
GPU_BATCH_TIMEOUT_MS = 50             # Faster batching (50ms)
GPU_THREAD_POOL_SIZE = 8              # More threads for GPU work
MAX_CONCURRENT_GPU_BATCHES = 4        # 4 parallel GPU batches = 128 concurrent

# Memory limits
MAX_AUDIO_BUFFER_BYTES = 5 * 1024 * 1024  # 5MB per session
AUDIO_CHUNK_TRIGGER_BYTES = 32000         # ~2 sec at 16kHz

# =============================================================================
# GLOBAL STATE
# =============================================================================

active_sessions: Dict[str, 'TranscriptionSession'] = {}
connection_count = 0
_lock = threading.Lock()

# Batching infrastructure
pending_jobs: asyncio.Queue = None          # Jobs waiting for batch
batch_semaphore: asyncio.Semaphore = None   # Limit concurrent batches
gpu_executor: ThreadPoolExecutor = None
_initialized = False


def init_vertical_scaling():
    """Initialize vertical scaling infrastructure"""
    global pending_jobs, batch_semaphore, gpu_executor, _initialized
    if _initialized:
        return
    
    pending_jobs = asyncio.Queue()
    batch_semaphore = asyncio.Semaphore(MAX_CONCURRENT_GPU_BATCHES)
    gpu_executor = ThreadPoolExecutor(
        max_workers=GPU_THREAD_POOL_SIZE,
        thread_name_prefix="gpu_worker"
    )
    _initialized = True
    print(f"[Vertical Scaling] Initialized: batch_size={GPU_BATCH_SIZE}, "
          f"gpu_threads={GPU_THREAD_POOL_SIZE}, max_connections={MAX_CONCURRENT_CONNECTIONS}")


@dataclass
class TranscriptionJob:
    """A single transcription job for batching"""
    session_id: str
    audio_data: bytes
    whisper_model: str
    language: str
    result_future: asyncio.Future = field(default_factory=lambda: asyncio.get_event_loop().create_future())


@dataclass  
class TranscriptionSession:
    """Session with optimized memory handling"""
    session_id: str
    whisper_model: str = "tiny"
    language: str = "en"
    audio_buffer: bytearray = field(default_factory=bytearray)
    chunk_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    transcription_results: list = field(default_factory=list)
    is_active: bool = True
    pending_job: Optional[TranscriptionJob] = None
    
    @property
    def buffer_mb(self) -> float:
        return len(self.audio_buffer) / (1024 * 1024)
    
    def add_audio_chunk(self, audio_data: bytes) -> bool:
        """Add chunk with memory limit check"""
        if len(self.audio_buffer) + len(audio_data) > MAX_AUDIO_BUFFER_BYTES:
            return False
        self.audio_buffer.extend(audio_data)
        self.chunk_count += 1
        self.last_activity = time.time()
        return True
    
    def extract_audio(self) -> bytes:
        """Extract and clear buffer"""
        audio = bytes(self.audio_buffer)
        self.audio_buffer.clear()
        return audio
    
    def has_enough_audio(self) -> bool:
        return len(self.audio_buffer) >= AUDIO_CHUNK_TRIGGER_BYTES


# =============================================================================
# BATCH PROCESSOR - Key to vertical scaling
# =============================================================================

async def batch_processor(get_model_func):
    """
    Background task that batches multiple transcription jobs together
    for efficient GPU utilization
    """
    global pending_jobs, batch_semaphore, gpu_executor
    
    print("[Batch Processor] Started")
    
    while True:
        try:
            # Collect batch
            batch: List[TranscriptionJob] = []
            
            # Wait for first job
            try:
                first_job = await asyncio.wait_for(
                    pending_jobs.get(),
                    timeout=1.0
                )
                batch.append(first_job)
            except asyncio.TimeoutError:
                continue
            
            # Try to fill batch (with timeout)
            batch_deadline = time.time() + (GPU_BATCH_TIMEOUT_MS / 1000)
            while len(batch) < GPU_BATCH_SIZE:
                remaining = batch_deadline - time.time()
                if remaining <= 0:
                    break
                try:
                    job = await asyncio.wait_for(
                        pending_jobs.get(),
                        timeout=remaining
                    )
                    batch.append(job)
                except asyncio.TimeoutError:
                    break
            
            # Process batch with semaphore to limit GPU concurrency
            async with batch_semaphore:
                await process_batch(batch, get_model_func)
                
        except Exception as e:
            print(f"[Batch Processor] Error: {e}")
            await asyncio.sleep(0.1)


async def process_batch(batch: List[TranscriptionJob], get_model_func):
    """Process a batch of transcription jobs on GPU"""
    if not batch:
        return
    
    # Group by model for efficient processing
    by_model: Dict[str, List[TranscriptionJob]] = {}
    for job in batch:
        if job.whisper_model not in by_model:
            by_model[job.whisper_model] = []
        by_model[job.whisper_model].append(job)
    
    # Process each model group
    for model_name, jobs in by_model.items():
        try:
            # Create temp files for all audio
            tmp_paths = []
            for job in jobs:
                tmp_path = _create_wav_file(job.audio_data)
                tmp_paths.append(tmp_path)
            
            # Get model
            model = get_model_func(model_name)
            
            # BATCH TRANSCRIPTION - single GPU call for multiple files!
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                gpu_executor,
                lambda: model.transcribe_with_vad(
                    tmp_paths,
                    lang_codes=[jobs[0].language] * len(jobs),
                    tasks=["transcribe"] * len(jobs),
                    batch_size=32,  # High batch for throughput
                )
            )
            
            # Distribute results
            for i, job in enumerate(jobs):
                try:
                    if i < len(results):
                        segments = results[i]
                        text = " ".join(s.get("text", "").strip() for s in segments)
                        job.result_future.set_result(text)
                    else:
                        job.result_future.set_result("")
                except Exception as e:
                    job.result_future.set_exception(e)
            
            # Cleanup
            for tmp_path in tmp_paths:
                try:
                    os.remove(tmp_path)
                except:
                    pass
                    
        except Exception as e:
            # Fail all jobs in this model group
            for job in jobs:
                if not job.result_future.done():
                    job.result_future.set_exception(e)


# =============================================================================
# WEBSOCKET HANDLER
# =============================================================================

async def handle_realtime_websocket(websocket: WebSocket, get_model_func):
    """Optimized WebSocket handler for vertical scaling"""
    global connection_count, pending_jobs
    
    # Connection limit check
    with _lock:
        if connection_count >= MAX_CONCURRENT_CONNECTIONS:
            await websocket.close(code=1013, reason="Server at capacity")
            return
        connection_count += 1
    
    await websocket.accept()
    session: Optional[TranscriptionSession] = None
    session_id = str(uuid.uuid4())
    
    try:
        while True:
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=CONNECTION_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                break
            
            message = json.loads(data)
            action = message.get("action", "")
            
            if action == "start":
                session = TranscriptionSession(
                    session_id=session_id,
                    whisper_model=message.get("whisper_model", "tiny"),
                    language=message.get("language", "en")
                )
                active_sessions[session_id] = session
                
                await websocket.send_json({
                    "status": "started",
                    "session_id": session_id
                })
                
            elif action == "audio" and session:
                try:
                    audio_bytes = base64.b64decode(message.get("data", ""))
                    
                    if not session.add_audio_chunk(audio_bytes):
                        await websocket.send_json({"warning": "buffer_full"})
                        continue
                    
                    # Check if ready to transcribe
                    if session.has_enough_audio() and session.pending_job is None:
                        audio = session.extract_audio()
                        
                        # Submit to batch queue
                        job = TranscriptionJob(
                            session_id=session_id,
                            audio_data=audio,
                            whisper_model=session.whisper_model,
                            language=session.language
                        )
                        session.pending_job = job
                        await pending_jobs.put(job)
                        
                        # Wait for result (non-blocking to other sessions)
                        try:
                            text = await asyncio.wait_for(
                                job.result_future,
                                timeout=30.0
                            )
                            if text:
                                session.transcription_results.append(text)
                                await websocket.send_json({
                                    "text": text,
                                    "is_final": False
                                })
                        except asyncio.TimeoutError:
                            await websocket.send_json({"error": "transcription_timeout"})
                        finally:
                            session.pending_job = None
                            
                except Exception as e:
                    await websocket.send_json({"error": str(e)})
                    
            elif action == "stop" and session:
                # Process remaining audio
                final_text = ""
                if len(session.audio_buffer) > 0:
                    audio = session.extract_audio()
                    job = TranscriptionJob(
                        session_id=session_id,
                        audio_data=audio,
                        whisper_model=session.whisper_model,
                        language=session.language
                    )
                    await pending_jobs.put(job)
                    try:
                        final_text = await asyncio.wait_for(job.result_future, timeout=30.0)
                        if final_text:
                            session.transcription_results.append(final_text)
                    except:
                        pass
                
                full_text = " ".join(session.transcription_results)
                await websocket.send_json({
                    "text": full_text,
                    "is_final": True
                })
                break
                
            elif action == "ping":
                await websocket.send_json({"action": "pong"})
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass
    finally:
        with _lock:
            connection_count -= 1
        if session_id in active_sessions:
            del active_sessions[session_id]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _create_wav_file(audio_data: bytes) -> str:
    """Create WAV from raw PCM (optimized - no unnecessary copies)"""
    sample_rate = 16000
    data_size = len(audio_data)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        # Write minimal WAV header
        tmp.write(b'RIFF')
        tmp.write(struct.pack('<I', 36 + data_size))
        tmp.write(b'WAVEfmt ')
        tmp.write(struct.pack('<IHHIIHH', 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
        tmp.write(b'data')
        tmp.write(struct.pack('<I', data_size))
        tmp.write(audio_data)
        return tmp.name


async def cleanup_stale_sessions():
    """Clean up timed-out sessions"""
    while True:
        await asyncio.sleep(60)
        now = time.time()
        stale = [
            sid for sid, s in list(active_sessions.items())
            if (now - s.last_activity) > CONNECTION_TIMEOUT_SECONDS
        ]
        for sid in stale:
            try:
                del active_sessions[sid]
            except:
                pass


# =============================================================================
# REGISTRATION
# =============================================================================

def register_websocket_routes(app, get_model_func):
    """Register routes and start background tasks"""
    
    init_vertical_scaling()
    
    @app.on_event("startup")
    async def startup():
        asyncio.create_task(batch_processor(get_model_func))
        asyncio.create_task(cleanup_stale_sessions())
        print("[Vertical Scaling] Background tasks started")
    
    @app.websocket("/ws/transcribe")
    async def ws_transcribe(websocket: WebSocket):
        await handle_realtime_websocket(websocket, get_model_func)
    
    @app.get("/ws/stats")
    async def get_stats():
        return {
            "connections": connection_count,
            "max_connections": MAX_CONCURRENT_CONNECTIONS,
            "active_sessions": len(active_sessions),
            "pending_jobs": pending_jobs.qsize() if pending_jobs else 0,
            "config": {
                "gpu_batch_size": GPU_BATCH_SIZE,
                "gpu_threads": GPU_THREAD_POOL_SIZE,
                "max_gpu_batches": MAX_CONCURRENT_GPU_BATCHES
            }
        }
    
    @app.get("/ws/sessions")
    async def get_sessions():
        return {
            "count": len(active_sessions),
            "sessions": [
                {"id": s.session_id, "chunks": s.chunk_count, "buffer_mb": round(s.buffer_mb, 2)}
                for s in list(active_sessions.values())[:100]
            ]
        }
