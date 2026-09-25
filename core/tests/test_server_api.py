"""
Unit tests for Emulator Server API Endpoints including Transcription Analysis.
"""

import json
from datetime import datetime
from unittest.mock import MagicMock

from engine.transcription_analyzer import TranscriptionAnalyzer
from server import EmulatorHandler


def test_server_detect_actions_api():
    # Test /api/detect_actions
    handler = MagicMock()
    # We can test transcription analyzer and server extraction directly
    from engine.transcription_analyzer import TranscriptionAnalyzer

    analyzer = TranscriptionAnalyzer()
    res = analyzer.analyze("I will call Rahul at 9 to review the launch roadmap.", use_gemini=False)

    assert len(res.potential_tasks) == 1
    task = res.potential_tasks[0]
    assert "Rahul" in task.task or "call" in task.task.lower()
    assert task.target_time_inferred_ampm in ["AM", "PM"]
    assert task.resolved_datetime is not None


def test_server_transcription_analysis_payload_structure():
    analyzer = TranscriptionAnalyzer()
    ref_dt = datetime(2026, 9, 12, 10, 0, 0)
    res = analyzer.analyze(
        "We have decided to ship the v2 release next week. Rahul, please send the release notes by Friday EOD.",
        ref_dt=ref_dt,
        use_gemini=False,
    )

    dict_repr = {
        "summary": res.summary,
        "topics": res.topics_discussed,
        "tone": res.sentiment_tone,
        "highlights": [
            {
                "headline": kh.headline,
                "takeaway": kh.takeaway,
                "speaker": kh.speaker,
                "category": kh.category,
                "importance": kh.importance,
                "quote": kh.verbatim_quote,
            }
            for kh in res.key_highlights
        ],
        "tasks": [
            {
                "owner": t.owner,
                "task": t.task,
                "category": t.category,
                "due": t.due_time_or_date,
                "resolved_datetime": t.resolved_datetime,
                "inferred_ampm": t.target_time_inferred_ampm,
                "quote": t.verbatim_quote,
                "urgency": t.urgency,
            }
            for t in res.potential_tasks
        ],
    }

    assert "summary" in dict_repr
    assert len(dict_repr["highlights"]) >= 1
    assert len(dict_repr["tasks"]) >= 1
    assert dict_repr["tasks"][0]["resolved_datetime"] is not None


def test_server_evaluate_payload_conversational_intelligence():
    """Verify that server evaluation pipeline produces complete conversational intelligence."""
    from engine.coaching_engine import ExecutiveCoachingEngine
    from engine.schema import ConversationSession, Utterance

    session = ConversationSession(
        session_id="test_server_eval",
        target_speaker="USER",
        counterpart_name="Rahul",
        counterpart_role="Collaborator",
        power_axis="LATERAL",
        dialogue=[
            Utterance(
                speaker="RAHUL", start_time=0.0, end_time=4.0, transcript="Hey Ashish, can we sync on deployment?"
            ),
            Utterance(
                speaker="USER", start_time=4.5, end_time=9.0, transcript="Yes Rahul, we agreed to ship on Thursday."
            ),
        ],
    )

    engine = ExecutiveCoachingEngine(use_local_only=True)
    evaluation = engine.evaluate_session(session)

    assert evaluation.dynamics is not None
    assert hasattr(evaluation.dynamics, "user_talk_time_pct")
    assert hasattr(evaluation.dynamics, "ask_vs_tell_ratio")
    assert hasattr(evaluation.dynamics, "brevity_potential_pct")
    assert hasattr(evaluation.dynamics, "deep_listening_score")

    assert isinstance(evaluation.emotional_trajectory, list)
    assert len(evaluation.emotional_trajectory) >= 1
    assert hasattr(evaluation.emotional_trajectory[0], "valence_score")
    assert hasattr(evaluation.emotional_trajectory[0], "tension_level")

    assert isinstance(evaluation.agreements, list)
    assert isinstance(evaluation.unresolved_loops, list)


def test_server_voiceprint_is_user_flag():
    """Verify that SpeakerVoiceprint distinguishes user from counterpart."""
    from asr_diarization.speaker_voiceprint_registry import SpeakerVoiceprint, SpeakerVoiceprintRegistry

    registry = SpeakerVoiceprintRegistry()
    user_vp = SpeakerVoiceprint(speaker_name="Ashish", role="App User", power_axis="SOLO", is_user=True)
    other_vp = SpeakerVoiceprint(speaker_name="Sandeep", role="VP of Eng", power_axis="UPWARD", is_user=False)

    registry.voiceprints["Ashish"] = user_vp
    registry.voiceprints["Sandeep"] = other_vp

    assert registry.voiceprints["Ashish"].is_user is True
    assert registry.voiceprints["Sandeep"].is_user is False
