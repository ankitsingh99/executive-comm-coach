"""
Automated unit tests for Acoustic Speaker Count & Vocal Tone Detection.
"""

import numpy as np
from asr_diarization.acoustic_speaker_detector import AcousticSpeakerToneDetector
from engine.schema import SpeakerAcousticProfile, AcousticAnalysisResult


def test_acoustic_detector_synthetic_single_speaker():
    detector = AcousticSpeakerToneDetector(sample_rate=16000)
    
    # Generate 2 seconds of synthetic 150 Hz tone (single voice)
    sr = 16000
    t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
    # Fundamental 150Hz with harmonics
    signal = (
        0.3 * np.sin(2 * np.pi * 150 * t) +
        0.15 * np.sin(2 * np.pi * 300 * t) +
        0.05 * np.sin(2 * np.pi * 450 * t)
    ).astype(np.float32)

    result = detector.analyze_audio_signal(signal, sample_rate=sr)

    assert isinstance(result, AcousticAnalysisResult)
    assert result.detected_speaker_count == 1
    assert result.is_multi_speaker is False
    assert len(result.speakers) == 1
    assert 130 <= result.speakers[0].mean_pitch_hz <= 170
    assert result.speakers[0].tone_label in ["Calm & Measured", "Monotone / Neutral", "Assertive & Decisive", "Natural & Conversational"]


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
