import os
from pathlib import Path

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "Audio Scam Detection API"
API_VERSION = "1.0.0"

# Audio Processing
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
CHUNK_DURATION = 10  # seconds

# Model Configuration
WHISPER_MODEL_SIZE = "medium"  # tiny, base, small, medium, large
AI_VOICE_MODEL = "as1605/Deepfake-audio-detection-V2"
# Using a more reliable emotion model that doesn't have tokenizer issues
EMOTION_MODEL = "facebook/wav2vec2-large-xlsr-53-english"  # More stable alternative
# Backup emotion model if needed: "superb/wav2vec2-base-superb-er"
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"

# LLM Configuration
OLLAMA_MODEL = "hermes3:8b"
LLM_TIMEOUT = 60  # seconds

# Directory Configuration
BASE_DIR = Path(__file__).parent
TEMP_AUDIO_DIR = BASE_DIR / "temp_audio"
CACHE_DIR = BASE_DIR / "cache"
SPEAKER_SEGMENTS_DIR = BASE_DIR / "speaker_segments"

# Hugging Face Token (must be set as environment variable)
HF_TOKEN = os.getenv("HF_TOKEN", "")
# Note: HF_TOKEN will be validated when models are actually loaded

# Processing Configuration
MAX_WORKERS = 4  # For parallel processing
AI_VOICE_THRESHOLD = 0.5
BACKGROUND_NOISE_THRESHOLD = 0.6
SCAM_CONFIDENCE_THRESHOLD = 70

# Cleanup Configuration
CLEANUP_TEMP_FILES = True
CACHE_RETENTION_DAYS = 7

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# CORS Configuration
ALLOWED_ORIGINS = ["*"]  # Configure appropriately for production
ALLOWED_METHODS = ["*"]
ALLOWED_HEADERS = ["*"]
