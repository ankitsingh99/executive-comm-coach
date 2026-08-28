"""
Unit tests for Gemini Multimodal Audio and Coaching Engine integration.
"""

import os
import pytest
from unittest.mock import MagicMock, patch
from asr_diarization.gemini_audio_engine import GeminiAudioEngine
from engine.gemini_coaching_engine import GeminiCoachingSynthesizer
from engine.coaching_engine import ExecutiveCoachingEngine
from engine.schema import ConversationSession, Utterance


def test_gemini_availability_without_key():
    with patch.dict(os.environ, {"GEMINI_API_KEY": "", "GOOGLE_API_KEY": ""}, clear=True):
        audio_engine = GeminiAudioEngine(api_key="")
        assert not audio_engine.is_available()

        coach_engine = GeminiCoachingSynthesizer(api_key="")
        assert not coach_engine.is_available()


def test_gemini_audio_engine_mock_transcription(tmp_path):
    fake_wav = tmp_path / "test_fake.wav"
    fake_wav.write_bytes(b"RIFF" + b"\x00" * 2000)

    audio_engine = GeminiAudioEngine(api_key="test_fake_api_key")

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '''{
      "transcription": [
        {
          "speaker": "USER",
          "start_time": 0.0,
          "end_time": 3.0,
          "transcript": "Ummm basically we should launch the product."
        }
      ],
      "speaker_count": 1,
      "overall_tone": "Calm & Measured",
      "speakers": [
        {
          "speaker_id": "SPEAKER_01",
          "tone_label": "Calm & Measured",
          "pitch_hz": 160.0,
          "talk_time_percentage": 100.0,
          "confidence_score": 0.98
        }
      ]
    }'''
    mock_client.models.generate_content.return_value = mock_response

    with patch.object(audio_engine, "_get_client", return_value=mock_client):
        audio_engine.api_key = "test_fake_api_key"
        utterances, acoustic_res = audio_engine.process_audio(str(fake_wav))

        assert len(utterances) == 1
        assert "basically" in utterances[0].transcript
        assert acoustic_res.detected_speaker_count == 1
        assert acoustic_res.overall_tone == "Calm & Measured"


def test_gemini_coaching_synthesizer_mock():
    synthesizer = GeminiCoachingSynthesizer(api_key="test_fake_api_key")

    session = ConversationSession(
        session_id="test_gemini_session",
        timestamp_utc="2026-08-28T00:00:00Z",
        target_speaker="USER",
        counterpart_name="VP of Eng",
        counterpart_role="VP of Eng",
        power_axis="UPWARD",
        dialogue=[
            Utterance(speaker="USER", start_time=0.0, end_time=4.0, transcript="Ummm I was thinking maybe we delay launch.")
        ]
    )

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '''{
      "persona_context": "Upward Executive Briefing",
      "metrics": {
        "presence_score": 75,
        "assertiveness_score": 70,
        "active_listening_score": 85,
        "filler_words_detected": [{"token": "ummm", "count": 1}]
      },
      "top_strengths": [
        {
          "observation": "Direct focus on launch timeline.",
          "verbatim_quote": "Ummm I was thinking maybe we delay launch."
        }
      ],
      "areas_for_improvement": [
        {
          "critique": "Hedging qualifiers ('maybe') reduce conviction. Action: State the recommendation directly as a decision.",
          "verbatim_quote": "maybe we delay launch",
          "coached_phrasing": "I recommend delaying the release to ensure full test coverage."
        }
      ],
      "longitudinal_summary": "Direct topical focus. Action: Eliminate qualifiers by leading with the bottom-line decision.",
      "persona_alignment_notes": "Evaluated against UPWARD (BLUF) communication rubric."
    }'''
    mock_client.models.generate_content.return_value = mock_response

    with patch.object(synthesizer, "_get_client", return_value=mock_client):
        synthesizer.api_key = "test_fake_api_key"
        evaluation = synthesizer.synthesize(session)

        assert evaluation is not None
        assert len(evaluation.top_strengths) == 1
        assert len(evaluation.areas_for_improvement) == 1
        assert "delaying the release" in evaluation.areas_for_improvement[0].coached_phrasing
