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
    mock_response.text = """{
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
    }"""
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
            Utterance(
                speaker="USER", start_time=0.0, end_time=4.0, transcript="Ummm I was thinking maybe we delay launch."
            )
        ],
    )

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = """{
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
    }"""
    mock_client.models.generate_content.return_value = mock_response

    with patch.object(synthesizer, "_get_client", return_value=mock_client):
        synthesizer.api_key = "test_fake_api_key"
        evaluation = synthesizer.synthesize(session)

        assert evaluation is not None
        assert len(evaluation.top_strengths) == 1
        assert len(evaluation.areas_for_improvement) == 1
        assert "delaying the release" in evaluation.areas_for_improvement[0].coached_phrasing


def test_gemini_coaching_synthesizer_top_n_and_error_handling():
    synthesizer = GeminiCoachingSynthesizer(api_key="test_fake_api_key")

    session = ConversationSession(
        session_id="test_top_n",
        target_speaker="USER",
        power_axis="INVALID_AXIS",
        dialogue=[Utterance(speaker="USER", start_time=0.0, end_time=2.0, transcript="Let's proceed.")],
    )

    mock_client = MagicMock()
    mock_response = MagicMock()
    # Markdown code fences wrapping json
    mock_response.text = """```json
    {
      "persona_context": "Direct",
      "metrics": {"presence_score": 80},
      "top_strengths": [{"observation": "Clear", "verbatim_quote": "Let's proceed."}],
      "areas_for_improvement": [{"critique": "Crisp", "verbatim_quote": "Let's proceed.", "coached_phrasing": "Proceed."}],
      "longitudinal_summary": "Good",
      "persona_alignment_notes": "Solo"
    }
    ```"""
    mock_client.models.generate_content.return_value = mock_response

    with patch.object(synthesizer, "_get_client", return_value=mock_client):
        synthesizer.api_key = "test_fake_api_key"
        eval_res = synthesizer.synthesize(session, top_n=1)
        assert eval_res is not None
        assert len(eval_res.top_strengths) == 1

        # Test exception fallback
        mock_client.models.generate_content.side_effect = Exception("API rate limited")
        assert synthesizer.synthesize(session) is None


def test_gemini_coaching_synthesizer_empty_actions_and_highlights_fallback():
    synthesizer = GeminiCoachingSynthesizer(api_key="test_fake_api_key")

    session = ConversationSession(
        session_id="test_fallback_actions",
        target_speaker="USER",
        power_axis="UPWARD",
        dialogue=[
            Utterance(speaker="USER", start_time=0.0, end_time=3.0, transcript="We have decided to ship by Friday."),
            Utterance(speaker="RAHUL", start_time=3.5, end_time=6.0, transcript="I will call Priya at 9.")
        ],
    )

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = """{
      "persona_context": "Upward Executive",
      "top_strengths": [{"observation": "Clear", "verbatim_quote": "We have decided."}],
      "areas_for_improvement": [{"critique": "Lead with BLUF", "verbatim_quote": "We have decided.", "coached_phrasing": "Ship Friday."}],
      "action_items": [],
      "key_highlights": []
    }"""
    mock_client.models.generate_content.return_value = mock_response

    with patch.object(synthesizer, "_get_client", return_value=mock_client):
        eval_res = synthesizer.synthesize(session)
        assert eval_res is not None
        # Verify fallback NLP populated actions and highlights
        assert len(eval_res.action_items) >= 1
        assert len(eval_res.key_highlights) >= 1


def test_gemini_client_initialization_failure_and_unavailable():
    """Test _get_client failure when genai.Client raises an error and is_available behavior."""
    synthesizer = GeminiCoachingSynthesizer(api_key="valid_key")
    with patch("google.genai.Client", side_effect=Exception("Failed to load genai")):
        client = synthesizer._get_client()
        assert client is None
        assert synthesizer.is_available() is False

    # Test synthesize returns None if not available or no client
    with patch.object(synthesizer, "is_available", return_value=False):
        assert synthesizer.synthesize(MagicMock()) is None

    with patch.object(synthesizer, "is_available", return_value=True), patch.object(synthesizer, "_get_client", return_value=None):
        assert synthesizer.synthesize(MagicMock()) is None


def test_gemini_synthesizer_function_calling_config_exception():
    """Test when AutomaticFunctionCallingConfig raises an exception."""
    synthesizer = GeminiCoachingSynthesizer(api_key="valid_key")
    session = ConversationSession(
        session_id="test_fc_exc",
        target_speaker="USER",
        dialogue=[Utterance(speaker="USER", start_time=0.0, end_time=2.0, transcript="Good morning.")]
    )
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '{"persona_context": "Test", "top_strengths": [], "areas_for_improvement": []}'
    mock_client.models.generate_content.return_value = mock_response

    with patch("google.genai.types.AutomaticFunctionCallingConfig", side_effect=TypeError("No such arg")), \
         patch.object(synthesizer, "_get_client", return_value=mock_client):
        res = synthesizer.synthesize(session)
        assert res is not None

