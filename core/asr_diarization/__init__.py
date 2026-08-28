"""ASR, Diarization, and Voiceprint Identification module."""
from .vad_gater import AmbientVadGate, VadFrameResult
from .sarvam_client import SarvamSpeechClient
from .diarizer import DiarizationEngine
from .local_stt_engine import LocalSTTEngine
from .nvidia_parakeet_engine import NvidiaParakeetEngine
from .acoustic_speaker_detector import AcousticSpeakerToneDetector
from .gemini_audio_engine import GeminiAudioEngine
from .speaker_voiceprint_registry import SpeakerVoiceprintRegistry, SpeakerVoiceprint

__all__ = [
    "AmbientVadGate",
    "VadFrameResult",
    "SarvamSpeechClient",
    "DiarizationEngine",
    "LocalSTTEngine",
    "NvidiaParakeetEngine",
    "AcousticSpeakerToneDetector",
    "GeminiAudioEngine",
    "SpeakerVoiceprintRegistry",
    "SpeakerVoiceprint"
]
