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


def test_filler_injection_relative_positions():
    """Test filler injection at start, middle, and end of utterance."""
    # 1. Beginning injection (rel_pos <= 0.25)
    u_start = Utterance(speaker="USER", start_time=0.0, end_time=4.0, transcript="we should ship today.")
    f_start = [AcousticFillerEvent(token="umm", start_time=0.1, end_time=0.6, duration_sec=0.5, speaker="USER")]
    inj_start = AcousticFillerDetector.inject_fillers_into_utterances([u_start], f_start)
    assert inj_start[0].transcript.startswith("umm")

    # 2. End injection (rel_pos >= 0.75)
    u_end = Utterance(speaker="USER", start_time=0.0, end_time=4.0, transcript="we should ship today.")
    f_end = [AcousticFillerEvent(token="aah", start_time=3.5, end_time=3.9, duration_sec=0.4, speaker="USER")]
    inj_end = AcousticFillerDetector.inject_fillers_into_utterances([u_end], f_end)
    assert inj_end[0].transcript.endswith("aah")

    # 3. Middle injection (0.25 < rel_pos < 0.75)
    u_mid = Utterance(speaker="USER", start_time=0.0, end_time=4.0, transcript="we should really ship today.")
    f_mid = [AcousticFillerEvent(token="uhh", start_time=2.0, end_time=2.5, duration_sec=0.5, speaker="USER")]
    inj_mid = AcousticFillerDetector.inject_fillers_into_utterances([u_mid], f_mid)
    assert "uhh" in inj_mid[0].transcript
    assert not inj_mid[0].transcript.startswith("uhh")
    assert not inj_mid[0].transcript.endswith("uhh")

    # 4. Invalid or missing inputs
    assert AcousticFillerDetector.inject_fillers_into_utterances([], f_start) == []
    assert AcousticFillerDetector.inject_fillers_into_utterances([u_start], []) == [u_start]

    # 5. Invalid WAV file path handling
    detector = AcousticFillerDetector()
    assert detector.detect_fillers_from_wav("/path/does/not/exist.wav") == []


def test_evaluate_segment_all_acoustic_types_and_speaker_mapping():
    """Test _evaluate_segment directly for 100% branch coverage."""
    detector = AcousticFillerDetector()

    # 1. Too short duration
    short_frames = [{"time": 0.0, "pitch_f0": 150.0, "centroid": 500.0, "low_ratio": 0.3, "mid_ratio": 0.3, "flux": 0.05}]
    assert detector._evaluate_segment(short_frames) is None

    # Helper to generate N frames of duration ~0.3s
    def make_frames(low=0.1, mid=0.1, centroid=1500.0, flux=0.05, n=12):
        return [
            {
                "time": i * 0.025,
                "pitch_f0": 150.0,
                "centroid": centroid,
                "low_ratio": low,
                "mid_ratio": mid,
                "flux": flux,
            }
            for i in range(n)
        ]

    # 2. 'uhh' token (1200 < centroid <= 1900, duration < 0.45)
    frames_uhh = make_frames(low=0.2, mid=0.2, centroid=1500.0, n=12)
    ev_uhh = detector._evaluate_segment(frames_uhh)
    assert ev_uhh is not None
    assert ev_uhh.token == "uhh"

    # 3. 'err' token (centroid > 1900, duration < 0.45)
    frames_err = make_frames(low=0.2, mid=0.2, centroid=2200.0, n=12)
    ev_err = detector._evaluate_segment(frames_err)
    assert ev_err is not None
    assert ev_err.token == "err"

    # 4. High spectral flux rejection
    frames_flux = make_frames(low=0.2, mid=0.2, centroid=1500.0, flux=0.8, n=12)
    assert detector._evaluate_segment(frames_flux) is None

    # 5. Speaker segment attribution
    utts = [Utterance(speaker="COUNTERPART", start_time=0.0, end_time=2.0, transcript="...")]
    ev_spk = detector._evaluate_segment(frames_uhh, speaker_segments=utts)
    assert ev_spk is not None
    assert ev_spk.speaker == "COUNTERPART"
