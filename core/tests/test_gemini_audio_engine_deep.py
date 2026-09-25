"""
Comprehensive test suite for GeminiAudioEngine covering timestamp parsing,
diarization normalization, Hinglish transcription, acoustic profile synthesis,
synthetic timestamp generation, and exception handling.
"""

import json
import os
import struct
import tempfile
import wave
from unittest.mock import patch, MagicMock
from asr_diarization.gemini_audio_engine import GeminiAudioEngine
from engine.schema import AcousticAnalysisResult


def create_mock_wav(duration_sec: float = 2.0, sample_rate: int = 16000) -> str:
    """Creates a temporary valid RIFF WAV file."""
    tf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    n_samples = int(duration_sec * sample_rate)
    with wave.open(tf.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        # write n_samples of silence/small noise
        raw_data = struct.pack(f"<{n_samples}h", *([100] * n_samples))
        wf.writeframes(raw_data)
    return tf.name


def test_gemini_audio_engine_availability_and_invalid_inputs():
    """Test availability checks and handling of missing or short files."""
    # 1. No API key
    engine_no_key = GeminiAudioEngine(api_key=None)
    assert not engine_no_key.is_available() or engine_no_key.api_key is not None

    # 2. Mock API key
    engine = GeminiAudioEngine(api_key="test_mock_gemini_key")
    with patch("google.genai.Client", return_value=MagicMock()):
        assert engine.is_available()

    # 3. Missing file
    utts, ac = engine.process_audio("/path/does/not/exist/audio.wav")
    assert utts == []
    assert isinstance(ac, AcousticAnalysisResult)

    # 4. File too short (< 1000 bytes)
    short_tf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    short_tf.write(b"RIFFshort")
    short_tf.close()
    try:
        with patch.object(engine, "is_available", return_value=True), patch.object(engine, "_get_client", return_value=MagicMock()):
            utts, ac = engine.process_audio(short_tf.name)
            assert utts == []
    finally:
        if os.path.exists(short_tf.name):
            os.remove(short_tf.name)


def test_gemini_audio_engine_success_multispeaker():
    """Test standard multi-speaker transcription with timestamp parsing variants."""
    wav_path = create_mock_wav(duration_sec=3.0)
    engine = GeminiAudioEngine(api_key="test_gemini_key")

    mock_gemini_response = {
        "transcription": [
            {
                "speaker": "SPEAKER 1",
                "start_time": "00:00:00",
                "end_time": "00:01",
                "transcript": "Dekho basically matlab latency is high."
            },
            {
                "speaker": "SPEAKER 2",
                "start_time": "00:00:00.8",
                "end_time": "00:02.5s",
                "transcript": "Haan, hum cache add kar sakte hain."
            },
            {
                "speaker": "DIRECTOR_ANAND",
                "start_time": "00:02",
                "end_time": "00:03",
                "transcript": "Approved. Proceed with rollout."
            }
        ],
        "speaker_count": 3,
        "overall_tone": "Strategic & Decisive",
        "speakers": [
            {"speaker_id": "USER", "pitch_hz": 160.0, "tone_label": "Inquiring", "talk_time_percentage": 40.0, "confidence_score": 0.96},
            {"speaker_id": "COUNTERPART", "pitch_hz": 190.0, "tone_label": "Collaborative", "talk_time_percentage": 40.0, "confidence_score": 0.94},
            {"speaker_id": "DIRECTOR_ANAND", "pitch_hz": 130.0, "tone_label": "Authoritative", "talk_time_percentage": 20.0, "confidence_score": 0.98}
        ]
    }

    mock_client = MagicMock()
    mock_resp = MagicMock()
    # Test markdown wrapping ```json ... ```
    mock_resp.text = f"```json\n{json.dumps(mock_gemini_response)}\n```"
    mock_client.models.generate_content.return_value = mock_resp

    try:
        with patch.object(engine, "_get_client", return_value=mock_client), \
             patch.object(engine, "is_available", return_value=True):
            utts, ac = engine.process_audio(wav_path)
            assert len(utts) == 3
            assert utts[0].speaker == "USER"
            assert utts[1].speaker == "COUNTERPART"
            assert utts[2].speaker == "DIRECTOR_ANAND"
            assert ac.detected_speaker_count == 3
            assert ac.is_multi_speaker is True
            assert len(ac.speakers) == 3
            assert ac.overall_tone == "Strategic & Decisive"
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


def test_gemini_audio_engine_zero_timestamps_and_fallback_profiles():
    """Test synthetic timestamp distribution when model returns zero/static timestamps."""
    wav_path = create_mock_wav(duration_sec=6.0)
    engine = GeminiAudioEngine(api_key="test_gemini_key")

    mock_gemini_response = {
        "transcription": [
            {
                "speaker": "SELF",
                "start_time": 0.0,
                "end_time": 0.0,
                "transcript": "First sentence spoken clearly."
            },
            {
                "speaker": "OTHER",
                "start_time": 0.0,
                "end_time": 0.0,
                "transcript": "Second sentence spoken clearly and with more detail."
            }
        ],
        "speaker_count": 2,
        "overall_tone": "Confident",
        "speakers": []  # Empty speakers to test fallback profile generation
    }

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = json.dumps(mock_gemini_response)
    mock_client.models.generate_content.return_value = mock_resp

    try:
        with patch.object(engine, "_get_client", return_value=mock_client), \
             patch.object(engine, "is_available", return_value=True):
            utts, ac = engine.process_audio(wav_path)
            assert len(utts) == 2
            # Verify timestamps were synthesized sequentially
            assert utts[0].start_time == 0.0
            assert utts[0].end_time > utts[0].start_time
            assert utts[1].start_time >= utts[0].end_time
            assert len(ac.speakers) == 1
            assert ac.speakers[0].speaker_id == "SPEAKER_01"
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


def test_gemini_audio_engine_exception_handling():
    """Test network or json decoding exceptions return empty results safely."""
    wav_path = create_mock_wav(duration_sec=2.0)
    engine = GeminiAudioEngine(api_key="test_gemini_key")

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("API Quota Exceeded")

    try:
        with patch.object(engine, "_get_client", return_value=mock_client), \
             patch.object(engine, "is_available", return_value=True):
            utts, ac = engine.process_audio(wav_path)
            assert utts == []
            assert isinstance(ac, AcousticAnalysisResult)
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)
