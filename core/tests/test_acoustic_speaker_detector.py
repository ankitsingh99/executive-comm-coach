"""
Automated unit tests for Acoustic Speaker Count & Vocal Tone Detection.
"""

import numpy as np
from asr_diarization.acoustic_speaker_detector import AcousticSpeakerToneDetector
from engine.schema import AcousticAnalysisResult, SpeakerAcousticProfile


def test_acoustic_detector_synthetic_single_speaker():
    detector = AcousticSpeakerToneDetector(sample_rate=16000)

    # Generate 2 seconds of synthetic 150 Hz tone (single voice)
    sr = 16000
    t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
    # Fundamental 150Hz with harmonics
    signal = (
        0.3 * np.sin(2 * np.pi * 150 * t) + 0.15 * np.sin(2 * np.pi * 300 * t) + 0.05 * np.sin(2 * np.pi * 450 * t)
    ).astype(np.float32)

    result = detector.analyze_audio_signal(signal, sample_rate=sr)

    assert isinstance(result, AcousticAnalysisResult)
    assert result.detected_speaker_count == 1
    assert result.is_multi_speaker is False
    assert len(result.speakers) == 1
    assert 130 <= result.speakers[0].mean_pitch_hz <= 170
    assert result.speakers[0].tone_label in [
        "Calm & Measured",
        "Monotone / Neutral",
        "Assertive & Decisive",
        "Natural & Conversational",
    ]


def test_acoustic_detector_synthetic_multi_speaker():
    detector = AcousticSpeakerToneDetector(sample_rate=16000)

    # Generate 4 seconds: 2s speaker A (130 Hz) + 2s speaker B (240 Hz)
    sr = 16000
    t1 = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
    t2 = np.linspace(2.0, 4.0, int(sr * 2.0), endpoint=False)

    spk_a = (0.35 * np.sin(2 * np.pi * 130 * t1) + 0.15 * np.sin(2 * np.pi * 260 * t1)).astype(np.float32)
    spk_b = (0.35 * np.sin(2 * np.pi * 240 * t2) + 0.15 * np.sin(2 * np.pi * 480 * t2)).astype(np.float32)
    signal = np.concatenate([spk_a, spk_b])

    result = detector.analyze_audio_signal(signal, sample_rate=sr)

    assert isinstance(result, AcousticAnalysisResult)
    assert result.detected_speaker_count == 2
    assert result.is_multi_speaker is True
    assert len(result.speakers) == 2
    # Verify both pitch baselines were captured
    pitches = [s.mean_pitch_hz for s in result.speakers]
    assert any(110 <= p <= 160 for p in pitches)
    assert any(210 <= p <= 270 for p in pitches)


def test_acoustic_detector_empty_and_short_audio():
    detector = AcousticSpeakerToneDetector(sample_rate=16000)

    # Empty signal
    res_empty = detector.analyze_audio_signal(np.array([], dtype=np.float32), sample_rate=16000)
    assert res_empty.detected_speaker_count == 1
    assert res_empty.is_multi_speaker is False

    # Near silent signal
    silence = np.zeros(16000, dtype=np.float32)
    res_silence = detector.analyze_audio_signal(silence, sample_rate=16000)
    assert res_silence.detected_speaker_count == 1


def test_acoustic_detector_analyze_wav_file(tmp_path):
    """Test analyze_wav_file with mono, stereo, different sample rates, and missing files."""
    import os
    import wave

    detector = AcousticSpeakerToneDetector(sample_rate=16000)

    # 1. Non-existent file
    res_missing = detector.analyze_wav_file("/path/not/found.wav")
    assert res_missing.detected_speaker_count == 1

    # 2. Valid mono WAV file at 16kHz
    wav_path = str(tmp_path / "valid_mono.wav")
    sr = 16000
    t = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False)
    signal = (0.3 * np.sin(2 * np.pi * 140 * t) + 0.15 * np.sin(2 * np.pi * 280 * t)).astype(np.float32)
    int16_audio = (signal * 32767).astype(np.int16)

    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(int16_audio.tobytes())

    res_wav = detector.analyze_wav_file(wav_path)
    assert res_wav.detected_speaker_count == 1
    assert len(res_wav.speakers) == 1

    # 3. Stereo WAV at 24kHz (tests multi-channel downmix and resampling)
    stereo_path = str(tmp_path / "stereo_24k.wav")
    sr_24k = 24000
    t_24k = np.linspace(0, 1.0, int(sr_24k * 1.0), endpoint=False)
    sig_l = (0.3 * np.sin(2 * np.pi * 150 * t_24k) * 32767).astype(np.int16)
    sig_r = (0.3 * np.sin(2 * np.pi * 150 * t_24k) * 32767).astype(np.int16)
    stereo_interleaved = np.empty((len(sig_l) + len(sig_r),), dtype=np.int16)
    stereo_interleaved[0::2] = sig_l
    stereo_interleaved[1::2] = sig_r

    with wave.open(stereo_path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr_24k)
        wf.writeframes(stereo_interleaved.tobytes())

    res_stereo = detector.analyze_wav_file(stereo_path)
    assert res_stereo.detected_speaker_count == 1

    # 4. 32-bit int and 8-bit uint WAV files
    wav_32 = str(tmp_path / "speaker_32.wav")
    samples_32 = (signal * 2147483647).astype(np.int32)
    with wave.open(wav_32, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(4)
        wf.setframerate(sr)
        wf.writeframes(samples_32.tobytes())
    assert detector.analyze_wav_file(wav_32).detected_speaker_count == 1

    wav_8 = str(tmp_path / "speaker_8.wav")
    samples_8 = np.clip((signal + 1.0) * 127.5, 0, 255).astype(np.uint8)
    with wave.open(wav_8, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(1)
        wf.setframerate(sr)
        wf.writeframes(samples_8.tobytes())
    assert detector.analyze_wav_file(wav_8).detected_speaker_count == 1


def test_acoustic_detector_all_tone_classifications():
    """Test every branch in _classify_tone directly."""
    detector = AcousticSpeakerToneDetector()

    # 1. Assertive & Decisive (mean_energy > 0.08 and pitch_std > 25.0)
    assert (
        detector._classify_tone(mean_pitch=150.0, pitch_range=60.0, mean_energy=0.10, pitch_std=28.0)
        == "Assertive & Decisive"
    )

    # 2. Dynamic & Expressive (pitch_std > 30.0 and pitch_range > 70.0)
    assert (
        detector._classify_tone(mean_pitch=160.0, pitch_range=80.0, mean_energy=0.05, pitch_std=32.0)
        == "Dynamic & Expressive"
    )

    # 3. Tense / Heightened (mean_pitch > 220.0 and pitch_std < 15.0)
    assert (
        detector._classify_tone(mean_pitch=240.0, pitch_range=30.0, mean_energy=0.05, pitch_std=10.0)
        == "Tense / Heightened"
    )

    # 4. Subdued & Reflective (mean_energy < 0.03 and pitch_std < 14.0)
    assert (
        detector._classify_tone(mean_pitch=140.0, pitch_range=20.0, mean_energy=0.02, pitch_std=11.0)
        == "Subdued & Reflective"
    )

    # 5. Monotone / Neutral (pitch_std < 10.0)
    assert (
        detector._classify_tone(mean_pitch=150.0, pitch_range=15.0, mean_energy=0.05, pitch_std=8.0)
        == "Monotone / Neutral"
    )

    # 6. Calm & Measured (mean_energy >= 0.03 and pitch_std >= 12.0)
    assert (
        detector._classify_tone(mean_pitch=150.0, pitch_range=30.0, mean_energy=0.05, pitch_std=15.0)
        == "Calm & Measured"
    )

    # 7. Natural & Conversational (fallback)
    assert (
        detector._classify_tone(mean_pitch=150.0, pitch_range=20.0, mean_energy=0.04, pitch_std=11.0)
        == "Natural & Conversational"
    )
