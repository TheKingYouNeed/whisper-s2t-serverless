"""
Test script for WhisperS2T Load Balancing Endpoint
"""
import requests
import time
import os

API_KEY = os.getenv("RUNPOD_API_KEY", "YOUR_API_KEY_HERE")
ENDPOINT_URL = "https://e5tlfqddk2uvn1.api.runpod.ai"

headers = {
    "Authorization": f"Bearer {API_KEY}"
}

print("=" * 60)
print("Testing WhisperS2T Load Balancing Endpoint")
print(f"URL: {ENDPOINT_URL}")
print("=" * 60)

# Test 1: Health check
print("\n1. Testing /health endpoint...")
try:
    response = requests.get(f"{ENDPOINT_URL}/health", headers=headers, timeout=60)
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
except Exception as e:
    print(f"   Error: {e}")

# Test 2: Root endpoint
print("\n2. Testing / root endpoint...")
try:
    response = requests.get(f"{ENDPOINT_URL}/", headers=headers, timeout=60)
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
except Exception as e:
    print(f"   Error: {e}")

# Test 3: Transcription with URL
print("\n3. Testing /transcribe_url with sample audio...")
audio_url = "https://github.com/runpod-workers/sample-inputs/raw/main/audio/gettysburg.wav"
print(f"   Audio URL: {audio_url}")

try:
    start_time = time.time()
    response = requests.post(
        f"{ENDPOINT_URL}/transcribe_url",
        headers=headers,
        data={"audio_url": audio_url},
        timeout=300
    )
    elapsed = time.time() - start_time
    print(f"   Status: {response.status_code}")
    print(f"   Time: {elapsed:.2f}s")
    if response.status_code == 200:
        result = response.json()
        print(f"   Transcription: {result.get('text', 'N/A')[:500]}...")
    else:
        print(f"   Response: {response.text[:500]}")
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "=" * 60)
print("Test complete!")
