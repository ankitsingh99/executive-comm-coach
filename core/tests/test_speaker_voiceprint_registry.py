"""
Unit tests for Persistent Speaker Voiceprint Memory & Acoustic Recognition Registry.
"""

import os

import numpy as np
import pytest
from asr_diarization.speaker_voiceprint_registry import SpeakerVoiceprint, SpeakerVoiceprintRegistry


@pytest.fixture
def temp_registry(tmp_path):
    storage_dir = tmp_path / "test_speaker_vault"
    return SpeakerVoiceprintRegistry(storage_dir=str(storage_dir))


def generate_synthetic_voice(
    pitch_f0: float = 140.0,
    duration_s: float = 1.0,
    sample_rate: int = 16000,
    timbre_formants: tuple = (500.0, 1500.0, 2500.0),
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
        name="Vikram Malhotra", role="VP of Engineering", power_axis="UPWARD", audio_signal_or_wav_path=voice_vikram
    )
    assert vp_vikram is not None
    assert vp_vikram.speaker_name == "Vikram Malhotra"
    assert vp_vikram.power_axis == "UPWARD"

    vp_pooja = temp_registry.enroll_speaker(
        name="Pooja Nair", role="Principal PM", power_axis="LATERAL", audio_signal_or_wav_path=voice_pooja
    )
    assert vp_pooja is not None
    assert vp_pooja.speaker_name == "Pooja Nair"
    assert vp_pooja.power_axis == "LATERAL"

    # 3. Test Recognition: Identify a new sample of Vikram's voice
    new_vikram_speech = generate_synthetic_voice(
        pitch_f0=127.0, duration_s=1.0, timbre_formants=(460.0, 1220.0, 2210.0)
    )
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


def test_multi_user_profiling_and_distinction(temp_registry):
    # Test that different app users (e.g. Ashish, Priya) and counterparts can be profiled & recognized
    user_ashish_voice = generate_synthetic_voice(
        pitch_f0=115.0, duration_s=1.0, timbre_formants=(400.0, 1100.0, 2100.0)
    )
    user_priya_voice = generate_synthetic_voice(pitch_f0=230.0, duration_s=1.0, timbre_formants=(750.0, 1900.0, 3200.0))
    counterpart_rohan = generate_synthetic_voice(
        pitch_f0=160.0, duration_s=1.0, timbre_formants=(520.0, 1400.0, 2400.0)
    )

    vp_ashish = temp_registry.enroll_speaker(
        name="Ashish", role="Self", power_axis="SOLO", audio_signal_or_wav_path=user_ashish_voice, is_user=True
    )
    assert vp_ashish.is_user is True

    vp_priya = temp_registry.enroll_speaker(
        name="Priya", role="Self", power_axis="SOLO", audio_signal_or_wav_path=user_priya_voice, is_user=True
    )
    assert vp_priya.is_user is True

    vp_rohan = temp_registry.enroll_speaker(
        name="Rohan", role="Tech Lead", power_axis="LATERAL", audio_signal_or_wav_path=counterpart_rohan, is_user=False
    )
    assert vp_rohan.is_user is False

    # Check recognition for User Ashish
    match_a = temp_registry.identify_speaker(user_ashish_voice, threshold=0.75)
    assert match_a is not None
    assert match_a[0].speaker_name == "Ashish"
    assert match_a[0].is_user is True

    # Check recognition for User Priya
    match_p = temp_registry.identify_speaker(user_priya_voice, threshold=0.75)
    assert match_p is not None
    assert match_p[0].speaker_name == "Priya"
    assert match_p[0].is_user is True

    # Check recognition for Counterpart Rohan
    match_r = temp_registry.identify_speaker(counterpart_rohan, threshold=0.75)
    assert match_r is not None
    assert match_r[0].speaker_name == "Rohan"
    assert match_r[0].is_user is False


def test_voiceprint_wav_file_enrollment_and_update(temp_registry, tmp_path):
    """Test enrolling directly from WAV files, updating existing profiles, and multi-channel handling."""
    import struct
    import wave

    # 1. Create 16-bit mono wav file
    wav_path = str(tmp_path / "speaker_test.wav")
    sr = 16000
    voice = generate_synthetic_voice(pitch_f0=150.0, duration_s=1.0, sample_rate=sr)
    int_samples = (voice * 32767).astype(np.int16)
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(int_samples.tobytes())

    # Enroll via wav path
    vp = temp_registry.enroll_speaker("Anand Sharma", "Director", "UPWARD", audio_signal_or_wav_path=wav_path)
    assert vp is not None
    assert vp.speaker_name == "Anand Sharma"
    assert vp.sample_count == 1

    # Update existing speaker with second wav sample (running average update)
    vp_updated = temp_registry.enroll_speaker(
        "Anand Sharma", "Senior Director", "UPWARD", audio_signal_or_wav_path=wav_path
    )
    assert vp_updated.sample_count == 2
    assert vp_updated.role == "Senior Director"

    # Identify via wav file path
    match = temp_registry.identify_speaker(wav_path, threshold=0.75)
    assert match is not None
    assert match[0].speaker_name == "Anand Sharma"

    # Test invalid audio path
    assert temp_registry.enroll_speaker("Ghost", audio_signal_or_wav_path="/invalid/path.wav") is None
    assert temp_registry.identify_speaker("/invalid/path.wav") is None
    assert temp_registry.enroll_speaker("Ghost", audio_signal_or_wav_path=12345) is None
    assert temp_registry.identify_speaker(12345) is None


def test_corrupted_database_recovery(tmp_path):
    """Test corrupted or invalid JSON database self-healing."""
    storage_dir = tmp_path / "corrupt_vault"
    storage_dir.mkdir(parents=True, exist_ok=True)
    db_file = storage_dir / "speaker_voiceprints.json"
    db_file.write_text("{ corrupt json ...")

    # Should recover gracefully without crashing
    reg = SpeakerVoiceprintRegistry(storage_dir=str(storage_dir))
    assert len(reg.list_enrolled_speakers()) == 0


def test_voiceprint_bitdepth_and_short_audio_branches(temp_registry, tmp_path):
    """Test 32-bit int, 8-bit uint WAV reading, short signals (<300ms), and deleting non-existent speaker."""
    import struct
    import wave

    sr = 16000
    voice = generate_synthetic_voice(pitch_f0=140.0, duration_s=1.0, sample_rate=sr)

    # 1. 32-bit int WAV
    wav_32 = str(tmp_path / "test_32bit.wav")
    samples_32 = (voice * 2147483647).astype(np.int32)
    with wave.open(wav_32, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(4)
        wf.setframerate(sr)
        wf.writeframes(samples_32.tobytes())

    vp32 = temp_registry.enroll_speaker("Rohan 32", "Eng", "LATERAL", audio_signal_or_wav_path=wav_32)
    assert vp32 is not None

    # 2. 8-bit uint WAV
    wav_8 = str(tmp_path / "test_8bit.wav")
    samples_8 = np.clip((voice + 1.0) * 127.5, 0, 255).astype(np.uint8)
    with wave.open(wav_8, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(1)
        wf.setframerate(sr)
        wf.writeframes(samples_8.tobytes())

    vp8 = temp_registry.enroll_speaker("Rohan 8", "Eng", "LATERAL", audio_signal_or_wav_path=wav_8)
    assert vp8 is not None

    # 3. Short signal extraction (< 300ms -> None)
    short_signal = np.ones(int(sr * 0.1), dtype=np.float32)
    assert temp_registry.extract_voiceprint_features(short_signal, sample_rate=sr) is None

    # 4. Silent signal (> 300ms but < 5 voiced frames -> None)
    silent_signal = np.zeros(int(sr * 0.5), dtype=np.float32)
    assert temp_registry.extract_voiceprint_features(silent_signal, sample_rate=sr) is None

    # 5. Deleting non-existent speaker
    assert temp_registry.delete_voiceprint("Non Existent Speaker") is False

    # 6. Synthesize fallback embedding directly
    fallback_emb = temp_registry.synthesize_fallback_embedding(150.0, 1200.0)
    assert fallback_emb.shape == (32,)

    # 7. Stereo WAV loading in _load_audio_file
    wav_stereo = str(tmp_path / "stereo_test.wav")
    samples_stereo = np.empty((8000, 2), dtype=np.int16)
    samples_stereo[:, 0] = (voice[:8000] * 32767).astype(np.int16)
    samples_stereo[:, 1] = (voice[:8000] * 32767).astype(np.int16)
    with wave.open(wav_stereo, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples_stereo.tobytes())

    sig, s_rate = temp_registry._load_audio_file(wav_stereo)
    assert s_rate == sr
    assert sig.ndim == 1

    # 8. Corrupted embedding vector self-healing during enroll update and identification
    from unittest.mock import patch

    temp_registry.enroll_speaker("Corrupt Speaker", "Engineer", "LATERAL", voice, sr)
    # Corrupt embedding vector to zeros
    temp_registry.voiceprints["Corrupt Speaker"].embedding_vector = [0.0] * 32

    # Updating speaker should self-heal via synthesize_fallback_embedding
    temp_registry.enroll_speaker("Corrupt Speaker", "Lead", "LATERAL", voice, sr)
    assert np.linalg.norm(temp_registry.voiceprints["Corrupt Speaker"].embedding_vector) > 0.5

    # Corrupt again and test identify_speaker self-healing
    temp_registry.voiceprints["Corrupt Speaker"].embedding_vector = [0.0] * 10
    identified = temp_registry.identify_speaker(voice, sample_rate=sr, threshold=0.1)
    assert identified is not None
    assert len(temp_registry.voiceprints["Corrupt Speaker"].embedding_vector) == 32
    assert np.linalg.norm(temp_registry.voiceprints["Corrupt Speaker"].embedding_vector) > 0.5

    # 9. save_to_disk exception handling
    with patch("builtins.open", side_effect=IOError("Disk permission denied")):
        temp_registry.save_to_disk()
