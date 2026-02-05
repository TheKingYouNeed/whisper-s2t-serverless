# Android Integration Guide for Real-Time Transcription

## WebSocket Protocol

### Connection
```
ws://YOUR_SERVER:PORT/ws/transcribe
```

### Message Flow

1. **Start Session**
```json
{"action": "start", "whisper_model": "tiny", "language": "en"}
```

2. **Send Audio Chunks** (every 1-2 seconds of audio)
```json
{"action": "audio", "data": "<base64 encoded PCM audio>"}
```

3. **Receive Transcriptions**
```json
{"text": "Hello world", "chunk_index": 1, "is_final": false}
```

4. **Stop Session**
```json
{"action": "stop"}
```

---

## Android (Kotlin) Example

```kotlin
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Base64
import okhttp3.*
import org.json.JSONObject
import java.util.concurrent.TimeUnit

class RealtimeTranscriber(
    private val serverUrl: String,
    private val whisperModel: String = "tiny",
    private val language: String = "en"
) {
    private var webSocket: WebSocket? = null
    private var audioRecord: AudioRecord? = null
    private var isRecording = false
    
    private val sampleRate = 16000
    private val channelConfig = AudioFormat.CHANNEL_IN_MONO
    private val audioFormat = AudioFormat.ENCODING_PCM_16BIT
    
    interface TranscriptionListener {
        fun onTranscription(text: String, isFinal: Boolean)
        fun onError(error: String)
        fun onConnected()
        fun onDisconnected()
    }
    
    private var listener: TranscriptionListener? = null
    
    fun setListener(listener: TranscriptionListener) {
        this.listener = listener
    }
    
    fun start() {
        val client = OkHttpClient.Builder()
            .readTimeout(0, TimeUnit.MILLISECONDS)
            .build()
        
        val request = Request.Builder()
            .url(serverUrl)
            .build()
        
        webSocket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                // Start session
                val startMsg = JSONObject().apply {
                    put("action", "start")
                    put("whisper_model", whisperModel)
                    put("language", language)
                }
                webSocket.send(startMsg.toString())
                listener?.onConnected()
                startRecording()
            }
            
            override fun onMessage(webSocket: WebSocket, text: String) {
                val json = JSONObject(text)
                when {
                    json.has("text") -> {
                        val transcription = json.getString("text")
                        val isFinal = json.optBoolean("is_final", false)
                        listener?.onTranscription(transcription, isFinal)
                    }
                    json.has("error") -> {
                        listener?.onError(json.getString("error"))
                    }
                }
            }
            
            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                listener?.onError(t.message ?: "Connection failed")
            }
            
            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                listener?.onDisconnected()
            }
        })
    }
    
    private fun startRecording() {
        val bufferSize = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat)
        
        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            sampleRate,
            channelConfig,
            audioFormat,
            bufferSize
        )
        
        isRecording = true
        audioRecord?.startRecording()
        
        Thread {
            val buffer = ByteArray(bufferSize)
            val chunkBuffer = ByteArrayOutputStream()
            val chunkDuration = 2000 // 2 seconds
            val bytesPerSecond = sampleRate * 2 // 16-bit = 2 bytes per sample
            val bytesPerChunk = bytesPerSecond * chunkDuration / 1000
            
            while (isRecording) {
                val read = audioRecord?.read(buffer, 0, buffer.size) ?: 0
                if (read > 0) {
                    chunkBuffer.write(buffer, 0, read)
                    
                    // Send chunk every ~2 seconds
                    if (chunkBuffer.size() >= bytesPerChunk) {
                        sendAudioChunk(chunkBuffer.toByteArray())
                        chunkBuffer.reset()
                    }
                }
            }
            
            // Send remaining audio
            if (chunkBuffer.size() > 0) {
                sendAudioChunk(chunkBuffer.toByteArray())
            }
        }.start()
    }
    
    private fun sendAudioChunk(audioData: ByteArray) {
        val base64Audio = Base64.encodeToString(audioData, Base64.NO_WRAP)
        val msg = JSONObject().apply {
            put("action", "audio")
            put("data", base64Audio)
        }
        webSocket?.send(msg.toString())
    }
    
    fun stop() {
        isRecording = false
        audioRecord?.stop()
        audioRecord?.release()
        audioRecord = null
        
        // Send stop command
        val stopMsg = JSONObject().apply {
            put("action", "stop")
        }
        webSocket?.send(stopMsg.toString())
    }
}
```

### Usage in Activity

```kotlin
class MainActivity : AppCompatActivity(), RealtimeTranscriber.TranscriptionListener {
    
    private lateinit var transcriber: RealtimeTranscriber
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        transcriber = RealtimeTranscriber(
            serverUrl = "ws://YOUR_SERVER:PORT/ws/transcribe",
            whisperModel = "tiny",
            language = "en"
        )
        transcriber.setListener(this)
    }
    
    fun startTranscription() {
        transcriber.start()
    }
    
    fun stopTranscription() {
        transcriber.stop()
    }
    
    override fun onTranscription(text: String, isFinal: Boolean) {
        runOnUiThread {
            transcriptionTextView.text = text
            if (isFinal) {
                // Final transcription received
            }
        }
    }
    
    override fun onError(error: String) {
        runOnUiThread {
            Toast.makeText(this, error, Toast.LENGTH_SHORT).show()
        }
    }
    
    override fun onConnected() {
        runOnUiThread {
            statusTextView.text = "Connected"
        }
    }
    
    override fun onDisconnected() {
        runOnUiThread {
            statusTextView.text = "Disconnected"
        }
    }
}
```

### Required Permissions (AndroidManifest.xml)
```xml
<uses-permission android:name="android.permission.RECORD_AUDIO" />
<uses-permission android:name="android.permission.INTERNET" />
```

### Dependencies (build.gradle)
```gradle
implementation 'com.squareup.okhttp3:okhttp:4.12.0'
```

---

## Python Test Client

```python
import asyncio
import base64
import json
import websockets

async def test_realtime():
    uri = "ws://1.208.108.242:33207/ws/transcribe"
    
    async with websockets.connect(uri) as ws:
        # Start session
        await ws.send(json.dumps({
            "action": "start",
            "whisper_model": "tiny",
            "language": "en"
        }))
        print("Session started:", await ws.recv())
        
        # Send test audio (read from file)
        with open("test_audio.wav", "rb") as f:
            audio_data = f.read()
        
        # Send in chunks
        chunk_size = 32000  # ~2 seconds at 16kHz
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i+chunk_size]
            await ws.send(json.dumps({
                "action": "audio",
                "data": base64.b64encode(chunk).decode()
            }))
            response = await ws.recv()
            print("Transcription:", response)
        
        # Stop session
        await ws.send(json.dumps({"action": "stop"}))
        final = await ws.recv()
        print("Final:", final)

asyncio.run(test_realtime())
```
