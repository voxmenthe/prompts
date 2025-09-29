source /Volumes/cdrive/repos/prompts/.venv/bin/activate
python src/data_processing/download_gsap_docs.py
python src/data_processing/download_livekit_docs.py
python src/data_processing/mirror_claude_api_docs.py
python src/data_processing/create_fastapi_mirror.py
