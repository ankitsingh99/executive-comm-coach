"""
Unit tests for AcousticFillerDetector, audio waveform hesitation parsing, and non-phonetic filler metrics.
"""

import numpy as np
import tempfile
import wave
import os

from asr_diarization.acoustic_filler_detector import AcousticFillerDetector
from engine.schema import Utterance, AcousticFillerEvent
from engine.metrics_calculator import MetricsCalculator


def _generate_synthetic_filler_wav(
    filler_type: str = "umm", duration_sec: float = 0.5, sample_rate: int = 16000
) -> str:
    """Generates synthetic audio containing a vocalized hesitation."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)

    if filler_type == "umm":
        # Low frequency nasal murmur ~180 Hz with weak harmonics and low amplitude
        signal = (
            0.15 * np.sin(2 * np.pi * 180 * t)
            + 0.05 * np.sin(2 * np.pi * 360 * t)
            + 0.01 * np.random.normal(0, 0.005, len(t))
        )
    elif filler_type == "aaaaa":
        # Prolonged vowel elongation ~220 Hz with open formants (660, 1100 Hz)
        signal = (
            0.20 * np.sin(2 * np.pi * 220 * t)
            + 0.12 * np.sin(2 * np.pi * 660 * t)
            + 0.08 * np.sin(2 * np.pi * 1100 * t)
            + 0.01 * np.random.normal(0, 0.005, len(t))
        )
    elif filler_type == "aah":
        # Shorter vowel hesitation ~240 Hz
        signal = (
            0.18 * np.sin(2 * np.pi * 240 * t)
            + 0.10 * np.sin(2 * np.pi * 720 * t)
            + 0.01 * np.random.normal(0, 0.005, len(t))
        )
    else:
        # Rapid dynamic modulated tone (simulating normal speech transitions)
        mod_freq = 150 + 80 * np.sin(2 * np.pi * 6 * t)
        signal = 0.20 * np.sin(2 * np.pi * mod_freq * t)

    # Pad with 0.2s silence before and after
    silence = np.zeros(int(sample_rate * 0.2), dtype=np.float32)
    full_audio = np.concatenate([silence, signal.astype(np.float32), silence])

    int16_audio = np.clip(full_audio * 32767, -32768, 32767).astype(np.int16)

    temp_wav = os.path.join(tempfile.gettempdir(), f"test_filler_{filler_type}_{int(duration_sec*1000)}.wav")
    with wave.open(temp_wav, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16_audio.tobytes())

    return temp_wav


def test_detect_nasal_murmur_umm():
    wav_path = _generate_synthetic_filler_wav("umm", duration_sec=0.5)
    detector = AcousticFillerDetector()
    events = detector.detect_fillers_from_wav(wav_path)

    if os.path.exists(wav_path):
        os.remove(wav_path)

    assert len(events) >= 1
    assert events[0].token in ["umm", "hmm"]
    assert events[0].duration_sec >= 0.25
    assert events[0].confidence >= 0.80


def test_detect_vowel_elongation_aaaaa():
    wav_path = _generate_synthetic_filler_wav("aaaaa", duration_sec=0.6)
    detector = AcousticFillerDetector()
    events = detector.detect_fillers_from_wav(wav_path)

    if os.path.exists(wav_path):
        os.remove(wav_path)

    assert len(events) >= 1
    assert events[0].token in ["aaaaa", "aah", "uhh"]
    assert events[0].duration_sec >= 0.35


def test_inject_fillers_into_utterances():
    utterances = [
        Utterance(speaker="USER", start_time=0.0, end_time=3.0, transcript="I was thinking about the proposal.")
    ]
    acoustic_fillers = [
        AcousticFillerEvent(token="umm", start_time=0.2, end_time=0.7, duration_sec=0.5, speaker="USER")
    ]

    injected = AcousticFillerDetector.inject_fillers_into_utterances(utterances, acoustic_fillers)
    assert len(injected) == 1
    assert "umm" in injected[0].transcript.lower()


def test_metrics_calculator_with_acoustic_fillers():
    utterances = [
        Utterance(speaker="USER", start_time=0.0, end_time=4.0, transcript="We will deploy the feature tomorrow.")
    ]
    acoustic_fillers = [
        AcousticFillerEvent(token="umm", start_time=0.5, end_time=1.0, duration_sec=0.5, speaker="USER"),
        AcousticFillerEvent(token="aah", start_time=2.0, end_time=2.4, duration_sec=0.4, speaker="USER"),
    ]

    metrics = MetricsCalculator.analyze_dialogue(utterances, target_speaker="USER", acoustic_fillers=acoustic_fillers)
    tokens = [f.token for f in metrics.filler_words_detected]
    assert "umm" in tokens
    assert "aah" in tokens
