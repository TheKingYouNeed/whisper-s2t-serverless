# 🎙️ Whisper AI Transcription API

**Fast, accurate, and affordable audio transcription powered by Whisper AI on high-performance GPUs.**

## ⚡ Key Features

- 🚀 **GPU Accelerated** - 50-130x real-time transcription speed
- 🔗 **Simple URL Input** - Just pass an audio URL, no file uploads needed
- 📁 **File Upload Support** - Or upload files directly
- 🎯 **Multiple Models** - Choose speed vs accuracy (tiny, small, large-v3)
- 🌍 **99+ Languages** - Transcribe audio in any language
- 📝 **Multiple Formats** - Get JSON, plain text, SRT, or VTT subtitles

---

## 🚀 Quick Start Examples

### Python - Transcribe from URL

```python
import requests

url = "https://whisper-api.p.rapidapi.com/transcribe_url"

payload = {
    "audio_url": "https://github.com/runpod-workers/sample-inputs/raw/main/audio/gettysburg.wav",
    "whisper_model": "tiny",
    "language": "en"
}

headers = {
    "X-RapidAPI-Key": "YOUR_RAPIDAPI_KEY",
    "X-RapidAPI-Host": "whisper-api.p.rapidapi.com",
    "Content-Type": "application/x-www-form-urlencoded"
}

response = requests.post(url, data=payload, headers=headers)
result = response.json()

print(result["text"])
```

### Python - Upload File

```python
import requests

url = "https://whisper-api.p.rapidapi.com/transcribe"

headers = {
    "X-RapidAPI-Key": "YOUR_RAPIDAPI_KEY",
    "X-RapidAPI-Host": "whisper-api.p.rapidapi.com"
}

files = {
    "file": open("audio.mp3", "rb")
}

data = {
    "whisper_model": "small",
    "language": "en"
}

response = requests.post(url, headers=headers, files=files, data=data)
result = response.json()

print(result["text"])
```

### JavaScript - Transcribe from URL

```javascript
const response = await fetch('https://whisper-api.p.rapidapi.com/transcribe_url', {
  method: 'POST',
  headers: {
    'X-RapidAPI-Key': 'YOUR_RAPIDAPI_KEY',
    'X-RapidAPI-Host': 'whisper-api.p.rapidapi.com',
    'Content-Type': 'application/x-www-form-urlencoded'
  },
  body: new URLSearchParams({
    audio_url: 'https://github.com/runpod-workers/sample-inputs/raw/main/audio/gettysburg.wav',
    whisper_model: 'tiny',
    language: 'en'
  })
});

const result = await response.json();
console.log(result.text);
```

### JavaScript - Upload File

```javascript
const formData = new FormData();
formData.append('file', audioFile);  // File object
formData.append('whisper_model', 'small');
formData.append('language', 'en');

const response = await fetch('https://whisper-api.p.rapidapi.com/transcribe', {
  method: 'POST',
  headers: {
    'X-RapidAPI-Key': 'YOUR_RAPIDAPI_KEY',
    'X-RapidAPI-Host': 'whisper-api.p.rapidapi.com'
  },
  body: formData
});

const result = await response.json();
console.log(result.text);
```

### cURL - Transcribe from URL

```bash
curl -X POST "https://whisper-api.p.rapidapi.com/transcribe_url" \
  -H "X-RapidAPI-Key: YOUR_RAPIDAPI_KEY" \
  -H "X-RapidAPI-Host: whisper-api.p.rapidapi.com" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "audio_url=https://github.com/runpod-workers/sample-inputs/raw/main/audio/gettysburg.wav" \
  -d "whisper_model=tiny" \
  -d "language=en"
```

### cURL - Upload File

```bash
curl -X POST "https://whisper-api.p.rapidapi.com/transcribe" \
  -H "X-RapidAPI-Key: YOUR_RAPIDAPI_KEY" \
  -H "X-RapidAPI-Host: whisper-api.p.rapidapi.com" \
  -F "file=@audio.mp3" \
  -F "whisper_model=small" \
  -F "language=en"
```

---

## 📝 API Endpoints

### POST /transcribe_url
Transcribe audio from a URL.

### POST /transcribe  
Transcribe an uploaded audio file.

---

## 📋 Parameters

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `audio_url` | ✅ Yes* | - | URL to audio file (for /transcribe_url) |
| `file` | ✅ Yes* | - | Audio file upload (for /transcribe) |
| `whisper_model` | No | tiny | Model: `tiny`, `tiny.en`, `small`, `large-v3` |
| `language` | No | en | Language code (en, es, fr, de, zh, ja, ko, etc.) |
| `output_format` | No | json | Output: `json`, `text`, `srt`, `vtt` |
| `batch_size` | No | 32 | Processing batch size (16-48) |
| `word_timestamps` | No | false | Include word-level timestamps |

---

## 🎯 Model Comparison

| Model | Speed | Accuracy | Best For |
|-------|-------|----------|----------|
| `tiny` | ⚡⚡⚡ 100x+ | Good | Real-time, quick drafts |
| `tiny.en` | ⚡⚡⚡ 130x+ | Good | English-only (fastest) |
| `small` | ⚡⚡ 90x+ | Better | Balanced quality/speed |
| `large-v3` | ⚡ 80x+ | Best | Final transcripts, accuracy |

---

## 📤 Response Format

### JSON Response (default)

```json
{
  "text": "Four score and seven years ago, our fathers brought forth on this continent a new nation...",
  "segments": [
    {
      "start": 0.0,
      "end": 3.5,
      "text": "Four score and seven years ago,"
    },
    {
      "start": 3.5,
      "end": 7.2,
      "text": "our fathers brought forth on this continent a new nation,"
    }
  ],
  "model": "tiny",
  "language": "en"
}
```

### Text Response (output_format=text)

```json
{
  "text": "Four score and seven years ago, our fathers brought forth on this continent a new nation..."
}
```

### SRT Response (output_format=srt)

```json
{
  "srt": "1\n00:00:00,000 --> 00:00:03,500\nFour score and seven years ago,\n\n2\n00:00:03,500 --> 00:00:07,200\nour fathers brought forth on this continent a new nation,"
}
```

---

## 🌍 Supported Languages

| Code | Language | Code | Language |
|------|----------|------|----------|
| en | English | zh | Chinese |
| es | Spanish | ja | Japanese |
| fr | French | ko | Korean |
| de | German | ru | Russian |
| it | Italian | ar | Arabic |
| pt | Portuguese | hi | Hindi |

Plus 90+ more languages! Use ISO 639-1 language codes.

---

## 💡 Use Cases

- 🎬 **Video Subtitles** - Generate SRT/VTT files instantly
- 🎙️ **Podcast Transcription** - Convert episodes to searchable text
- 📞 **Call Analytics** - Transcribe customer calls for insights
- 📚 **Content Creation** - Turn audio into blog posts
- 🎓 **Education** - Transcribe lectures and meetings
- 🏥 **Medical** - Transcribe doctor-patient conversations
- ⚖️ **Legal** - Transcribe depositions and court recordings

---

## ⚠️ Limits & Supported Formats

**File Limits:**
- Max file size: 100MB
- Max audio length: 2 hours

**Supported Audio Formats:**
- MP3, WAV, M4A, FLAC, OGG, WEBM, AAC, WMA

---

## ❓ FAQ

**Q: Which model should I use?**  
A: Start with `tiny` for speed. Use `large-v3` for final, accurate transcripts.

**Q: How do I transcribe non-English audio?**  
A: Set the `language` parameter to the appropriate ISO code (e.g., `es` for Spanish).

**Q: Can I get subtitles for my video?**  
A: Yes! Set `output_format=srt` or `output_format=vtt` to get subtitle files.

**Q: How fast is the transcription?**  
A: With GPU acceleration, we transcribe at 50-130x real-time speed. A 2-minute audio file takes about 1-2 seconds!

---

## 🆘 Support

Questions or issues? Contact us through RapidAPI messaging.
