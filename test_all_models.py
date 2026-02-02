"""Test all available Whisper models on the RunPod endpoint"""
import requests
import time
import os

API_KEY = os.getenv("RUNPOD_API_KEY", "YOUR_API_KEY_HERE")
ENDPOINT_URL = "https://e5tlfqddk2uvn1.api.runpod.ai"

headers = {"Authorization": f"Bearer {API_KEY}"}

# Sample audio URL
audio_url = "https://github.com/runpod-workers/sample-inputs/raw/main/audio/gettysburg.wav"

# 4 models available in the image
MODELS = ['tiny', 'tiny.en', 'small', 'large-v3']

print("=" * 70)
print("Testing ALL Whisper models on RunPod endpoint")
print(f"Audio: {audio_url}")
print("=" * 70)

for model in MODELS:
    print(f"\n[{model}] Testing...")
    try:
        start = time.time()
        resp = requests.post(
            f"{ENDPOINT_URL}/transcribe_url",
            headers=headers,
            data={"audio_url": audio_url, "model": model},
            timeout=180
        )
        elapsed = time.time() - start
        
        if resp.status_code == 200:
            result = resp.json()
            text = result.get('text', '')[:100]
            print(f"    ✅ Status: 200 | Time: {elapsed:.1f}s")
            print(f"    Text: {text}...")
        else:
            print(f"    ❌ Status: {resp.status_code}")
            print(f"    Error: {resp.text[:200]}")
    except Exception as e:
        print(f"    ❌ Error: {e}")

print("\n" + "=" * 70)
print("All models tested!")
