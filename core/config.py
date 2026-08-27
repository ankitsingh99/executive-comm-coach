"""
Local On-Device Configuration for Executive Communication Coach.
Enforces zero-cloud offline processing during development.
"""

import os
from enum import Enum


class ProcessingMode(str, Enum):
    LOCAL_ON_DEVICE = "LOCAL_ON_DEVICE"
    CLOUD_HYBRID = "CLOUD_HYBRID"


# Default to 100% local on-device execution
CURRENT_PROCESSING_MODE = ProcessingMode.LOCAL_ON_DEVICE

# Local directories
DATA_DIR = os.path.expanduser("~/.exec_coach_local")
ENCRYPTED_STORAGE_DIR = os.path.join(DATA_DIR, "encrypted_vault")
SESSIONS_DB_PATH = os.path.join(DATA_DIR, "coaching_sessions.json")

# Local LLM settings (optional Ollama / LiteRT local endpoint)
LOCAL_OLLAMA_URL = os.getenv("LOCAL_OLLAMA_URL", "http://localhost:11434")
LOCAL_OLLAMA_MODEL = os.getenv("LOCAL_OLLAMA_MODEL", "gemma2:2b")

# Ensure local directories exist
os.makedirs(ENCRYPTED_STORAGE_DIR, exist_ok=True)
