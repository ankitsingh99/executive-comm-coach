"""
Unit tests for Persistent Speaker Voiceprint Memory & Acoustic Recognition Registry.
"""

import os
import pytest
import numpy as np
from asr_diarization.speaker_voiceprint_registry import SpeakerVoiceprintRegistry, SpeakerVoiceprint


@pytest.fixture
def temp_registry(tmp_path):
    storage_dir = tmp_path / "test_speaker_vault"
    return SpeakerVoiceprintRegistry(storage_dir=str(storage_dir))


def generate_synthetic_voice(
    pitch_f0: float = 140.0,
    duration_s: float = 1.0,
    sample_rate: int = 16000,
    timbre_formants: tuple = (500.0, 1500.0, 2500.0)
) -> np.ndarray:
    """Generates a synthetic voice harmonic waveform with defined pitch and formants."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    # Fundamental harmonic
    signal = 0.4 * np.sin(2 * np.pi * pitch_f0 * t)
    signal += 0.25 * np.sin(2 * np.pi * (2 * pitch_f0) * t)
    signal += 0.15 * np.sin(2 * np.pi * (3 * pitch_f0) * t)

    # Formant resonances
    for f in timbre_formants:
        signal += 0.1 * np.sin(2 * np.pi * f * t)

    # Add minor envelope modulation to simulate natural speech syllables
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t)
    signal = signal * env

    # Normalize
    signal = signal / (np.max(np.abs(signal)) + 1e-6)
    return signal.astype(np.float32)


def test_feature_extraction(temp_registry):
    voice = generate_synthetic_voice(pitch_f0=160.0, duration_s=1.0)
    features = temp_registry.extract_voiceprint_features(voice, sample_rate=16000)

    assert features is not None
    emb, mean_pitch, pitch_range, centroid = features
    assert len(emb) == 32
    assert 130.0 <= mean_pitch <= 190.0
    assert centroid > 100.0


def test_voiceprint_enrollment_and_identification(temp_registry):
    # 1. Generate distinct voices for two counterparts
    # Voice A: Vikram (Deep executive voice, pitch ~125 Hz)
    voice_vikram = generate_synthetic_voice(pitch_f0=125.0, duration_s=1.2, timbre_formants=(450.0, 1200.0, 2200.0))
    # Voice B: Pooja (Dynamic product manager voice, pitch ~220 Hz)
    voice_pooja = generate_synthetic_voice(pitch_f0=220.0, duration_s=1.2, timbre_formants=(700.0, 1800.0, 3100.0))

    # 2. Enroll both voices
    vp_vikram = temp_registry.enroll_speaker(
        name="Vikram Malhotra",
        role="VP of Engineering",
        power_axis="UPWARD",
        audio_signal_or_wav_path=voice_vikram
    )
    assert vp_vikram is not None
    assert vp_vikram.speaker_name == "Vikram Malhotra"
    assert vp_vikram.power_axis == "UPWARD"

    vp_pooja = temp_registry.enroll_speaker(
        name="Pooja Nair",
        role="Principal PM",
        power_axis="LATERAL",
        audio_signal_or_wav_path=voice_pooja
    )
    assert vp_pooja is not None
    assert vp_pooja.speaker_name == "Pooja Nair"
    assert vp_pooja.power_axis == "LATERAL"

    # 3. Test Recognition: Identify a new sample of Vikram's voice
    new_vikram_speech = generate_synthetic_voice(pitch_f0=127.0, duration_s=1.0, timbre_formants=(460.0, 1220.0, 2210.0))
    match_result = temp_registry.identify_speaker(new_vikram_speech, threshold=0.75)
    assert match_result is not None
    matched_vp, confidence = match_result
    assert matched_vp.speaker_name == "Vikram Malhotra"
    assert matched_vp.role == "VP of Engineering"
    assert matched_vp.power_axis == "UPWARD"
    assert confidence >= 0.75

    # 4. Test Recognition: Identify a new sample of Pooja's voice
    new_pooja_speech = generate_synthetic_voice(pitch_f0=218.0, duration_s=1.0, timbre_formants=(690.0, 1790.0, 3090.0))
    match_result_p = temp_registry.identify_speaker(new_pooja_speech, threshold=0.75)
    assert match_result_p is not None
    matched_vp_p, confidence_p = match_result_p
    assert matched_vp_p.speaker_name == "Pooja Nair"
    assert matched_vp_p.power_axis == "LATERAL"
    assert confidence_p >= 0.75


def test_voiceprint_rejection_and_erasure(temp_registry):
    voice_vikram = generate_synthetic_voice(pitch_f0=120.0, duration_s=1.0)
    temp_registry.enroll_speaker("Vikram Malhotra", "VP", "UPWARD", voice_vikram)

    # Unknown high pitch voice
    unknown_voice = generate_synthetic_voice(pitch_f0=350.0, duration_s=1.0, timbre_formants=(1200.0, 3000.0, 4200.0))
    match_unknown = temp_registry.identify_speaker(unknown_voice, threshold=0.85)
    assert match_unknown is None

    # Test Listing
    enrolled = temp_registry.list_enrolled_speakers()
    assert len(enrolled) == 1
    assert enrolled[0]["name"] == "Vikram Malhotra"

    # Test DPDP Right to Erasure
    deleted = temp_registry.delete_voiceprint("Vikram Malhotra")
    assert deleted is True
    assert len(temp_registry.list_enrolled_speakers()) == 0
    assert temp_registry.identify_speaker(voice_vikram) is None
