"""
Unit tests for SarvamSpeechClient covering multipart payload building,
API mocking, diarization parsing, fallback transcripts, and error handling.
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

from asr_diarization.sarvam_client import SarvamSpeechClient


def test_sarvam_client_availability():
    """Verify availability check based on API key presence."""
    client_empty = SarvamSpeechClient(api_key="")
    assert not client_empty.is_available()

    client_with_key = SarvamSpeechClient(api_key="sarvam_mock_key_123")
    assert client_with_key.is_available()


def test_sarvam_multipart_builder():
    """Verify multipart form-data payload generation."""
    client = SarvamSpeechClient(api_key="mock_key")
    content_type, body = client._build_multipart_payload(
        fields={"model": "saaras:v2", "language_code": "hi-IN"},
        file_field="file",
        filename="test.wav",
        file_bytes=b"RIFFdummywavbytes",
    )
    assert "multipart/form-data; boundary=" in content_type
    assert b'Content-Disposition: form-data; name="model"' in body
    assert b"saaras:v2" in body
    assert b"RIFFdummywavbytes" in body


def test_sarvam_parse_diarized_entries_multi_speaker():
    """Verify parsing of Sarvam diarized entries mapping to USER, COUNTERPART, and SPEAKER_03."""
    client = SarvamSpeechClient(api_key="mock_key")
    sample_response = {
        "diarized_transcript": {
            "entries": [
                {
                    "speaker_id": "spk_0",
                    "transcript": "Namaste, kaise hain aap?",
                    "start_time_seconds": 0.0,
                    "end_time_seconds": 2.5,
                },
                {
                    "speaker_id": "spk_1",
                    "transcript": "Main theek hoon, review shuru karein?",
                    "start_time_seconds": 3.0,
                    "end_time_seconds": 5.8,
                },
                {
                    "speaker_id": "spk_2",
                    "transcript": "Haan main bhi aligned hoon.",
                    "start_time_seconds": 6.0,
                    "end_time_seconds": 8.0,
                },
            ]
        }
    }
    utterances = client._parse_sarvam_response(sample_response)
    assert len(utterances) == 3
    assert utterances[0].speaker == "USER"
    assert utterances[0].transcript == "Namaste, kaise hain aap?"
    assert utterances[1].speaker == "COUNTERPART"
    assert utterances[2].speaker == "SPEAKER_03"


def test_sarvam_parse_fallback_transcript():
    """Verify fallback when diarized entries are absent."""
    client = SarvamSpeechClient(api_key="mock_key")
    fallback_response = {"transcript": "Aaj hum deployment schedule discuss karenge."}
    utterances = client._parse_sarvam_response(fallback_response)
    assert len(utterances) == 1
    assert utterances[0].speaker == "USER"
    assert utterances[0].transcript == "Aaj hum deployment schedule discuss karenge."


def test_sarvam_transcribe_audio_chunk_mock():
    """Test full transcribe_audio_chunk workflow with mocked HTTP response."""
    client = SarvamSpeechClient(api_key="valid_test_key")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        tf.write(b"RIFFwavmock")
        temp_wav = tf.name

    try:
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.read.return_value = json.dumps(
            {
                "diarized_transcript": {
                    "entries": [
                        {
                            "speaker_id": "spk_0",
                            "transcript": "Hello team",
                            "start_time_seconds": 0.0,
                            "end_time_seconds": 2.0,
                        }
                    ]
                }
            }
        ).encode("utf-8")

        with patch("urllib.request.urlopen", return_value=mock_resp):
            utts = client.transcribe_audio_chunk(temp_wav)
            assert len(utts) == 1
            assert utts[0].transcript == "Hello team"

        # Test HTTP non-200 error
        mock_resp_err = MagicMock()
        mock_resp_err.status = 401
        mock_resp_err.__enter__.return_value = mock_resp_err
        with patch("urllib.request.urlopen", return_value=mock_resp_err):
            assert client.transcribe_audio_chunk(temp_wav) == []

        # Test network exception
        with patch("urllib.request.urlopen", side_effect=Exception("Timeout")):
            assert client.transcribe_audio_chunk(temp_wav) == []

    finally:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
