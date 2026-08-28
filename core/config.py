"""
Local On-Device Configuration for Executive Communication Coach.
Enforces zero-cloud offline processing during development.
"""

import os
from enum import Enum


class ProcessingMode(str, Enum):
    LOCAL_ON_DEVICE = "LOCAL_ON_DEVICE"
    CLOUD_HYBRID = "CLOUD_HYBRID"


# Load .env if present
ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
if os.path.exists(ENV_PATH):
    try:
        with open(ENV_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k and not os.getenv(k):
                        os.environ[k] = v
    except Exception:
        pass


def get_gemini_api_key() -> str:
    """Retrieves Gemini API Key from environment or .env file."""
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""


# Gemini Configuration
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_KEY = get_gemini_api_key()

# Default processing mode: CLOUD_HYBRID if GEMINI_API_KEY available, else LOCAL_ON_DEVICE
CURRENT_PROCESSING_MODE = ProcessingMode.CLOUD_HYBRID if GEMINI_API_KEY else ProcessingMode.LOCAL_ON_DEVICE

# Local directories
DATA_DIR = os.path.expanduser("~/.exec_coach_local")
ENCRYPTED_STORAGE_DIR = os.path.join(DATA_DIR, "encrypted_vault")
SESSIONS_DB_PATH = os.path.join(DATA_DIR, "coaching_sessions.json")

# Local LLM settings (optional Ollama fallback)
LOCAL_OLLAMA_URL = os.getenv("LOCAL_OLLAMA_URL", "http://localhost:11434")
LOCAL_OLLAMA_MODEL = os.getenv("LOCAL_OLLAMA_MODEL", "gemma2:2b")

# Ensure local directories exist
os.makedirs(ENCRYPTED_STORAGE_DIR, exist_ok=True)
