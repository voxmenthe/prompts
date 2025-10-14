source /Volumes/cdrive/repos/prompts/.venv/bin/activate
python src/DATA_SCRIPTS/download_gsap_docs.py
python src/DATA_SCRIPTS/download_livekit_docs.py
python src/DATA_SCRIPTS/mirror_claude_api_docs.py
# python src/DATA_SCRIPTS/mirror_openai_api_docs.py
python src/DATA_SCRIPTS/create_fastapi_mirror.py
