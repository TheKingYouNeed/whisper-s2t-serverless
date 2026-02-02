"""Download CTranslate2 Whisper models during Docker build (no CUDA needed)"""
from huggingface_hub import snapshot_download

MODELS = ['tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'large-v3']

for i, m in enumerate(MODELS, 1):
    print(f'{i}/{len(MODELS)} Downloading {m}...')
    repo = f'Systran/faster-whisper-{m}'
    snapshot_download(repo_id=repo)
    
print('All models downloaded!')
