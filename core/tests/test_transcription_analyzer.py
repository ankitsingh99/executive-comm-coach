"""
Unit tests for Transcription Analyzer Engine (Key Highlights & Potential Tasks).
"""

from datetime import datetime
from engine.schema import Utterance, ConversationSession
from engine.transcription_analyzer import TranscriptionAnalyzer
from engine.action_item_extractor import ActionItemExtractor


def test_transcription_analyzer_key_highlights():
    dialogue = [
        Utterance(
            speaker="Rahul",
            start_time=0.0,
            end_time=4.0,
            transcript="We have decided to migrate our entire caching layer to Redis cluster next sprint.",
        ),
        Utterance(
            speaker="USER",
            start_time=4.5,
            end_time=8.0,
            transcript="The core objective is to decrease p99 query latency under 50 milliseconds.",
        ),
        Utterance(
            speaker="Rahul",
            start_time=8.5,
            end_time=12.0,
            transcript="The major blocker is legacy connection pooling, so we must be cautious.",
        ),
    ]

    analyzer = TranscriptionAnalyzer()
    highlights = analyzer.extract_key_highlights(dialogue)

    assert len(highlights) >= 2
    categories = [h.category for h in highlights]
    assert "Decision" in categories or "Strategy" in categories or "Key Insight" in categories
    assert any("redis" in h.takeaway.lower() or "latency" in h.takeaway.lower() for h in highlights)


def test_transcription_analyzer_potential_tasks_with_time_resolution():
    # Set ref_dt to 11:30 AM
    ref_dt = datetime(2026, 9, 12, 11, 30, 0)
    dialogue = [
        Utterance(
            speaker="Rahul",
            start_time=0.0,
            end_time=4.0,
            transcript="I will call xyz at 9 to finalize the architecture.",
        ),
        Utterance(
            speaker="USER",
            start_time=4.5,
            end_time=8.0,
            transcript="We have decided to ship the release by tomorrow morning.",
        ),
    ]

    analyzer = TranscriptionAnalyzer()
    tasks = analyzer.extract_potential_tasks(dialogue, ref_dt=ref_dt)

    assert len(tasks) == 2
    call_task = next(t for t in tasks if "call" in t.task.lower() or "xyz" in t.task.lower())
    assert call_task.owner == "Rahul"
    assert call_task.target_time_inferred_ampm == "PM"
    assert "2026-09-12T21:00:00" in call_task.resolved_datetime


def test_transcription_analyzer_full_analysis():
    raw_transcript = (
        "Rahul: We have decided to launch the enterprise beta this month.\n"
        "USER: Great. I will call Priya at 9 to coordinate the announcement.\n"
        "Rahul: Please make sure to send the release notes by Friday EOD."
    )

    analyzer = TranscriptionAnalyzer()
    ref_dt = datetime(2026, 9, 12, 8, 30, 0)
    result = analyzer.analyze(raw_transcript, ref_dt=ref_dt, use_gemini=False)

    assert len(result.key_highlights) >= 1
    assert len(result.potential_tasks) >= 2
    assert len(result.topics_discussed) >= 1
    assert result.summary != ""
    assert result.sentiment_tone != ""

    # Verify the "call Priya at 9" at 8:30 AM resolved to 9:00 AM today (inferred AM)
    priya_task = next(t for t in result.potential_tasks if "priya" in t.task.lower() or "call" in t.task.lower())
    assert priya_task.target_time_inferred_ampm == "AM"
    assert "2026-09-12T09:00:00" in priya_task.resolved_datetime


def test_transcription_analyzer_gemini_mock():
    """Test LLM-based analysis with Gemini response mocking and fallback."""
    import json
    from unittest.mock import patch, MagicMock

    dialogue = [
        Utterance(speaker="USER", start_time=0.0, end_time=4.0, transcript="We will deploy the caching subsystem on Friday.")
    ]

    mock_llm_json = {
        "summary": "The team aligned on the upcoming deployment schedule.",
        "key_highlights": [
            {
                "headline": "Caching Deployment Scheduled",
                "takeaway": "Subsystem is ready for Friday release.",
                "speaker": "USER",
                "verbatim_quote": "We will deploy the caching subsystem on Friday.",
                "category": "Milestone",
                "importance": "High"
            }
        ],
        "potential_tasks": [
            {
                "owner": "USER",
                "task": "Deploy the caching subsystem",
                "due_time_or_date": "Friday",
                "resolved_datetime": "2026-09-18T17:00:00Z",
                "target_time_inferred_ampm": "PM",
                "verbatim_quote": "We will deploy the caching subsystem on Friday.",
                "category": "Deliverable / Commitment",
                "urgency": "High"
            }
        ],
        "topics_discussed": ["Caching Subsystem", "Deployment Schedule"],
        "sentiment_tone": "Decisive & Collaborative"
    }

    analyzer = TranscriptionAnalyzer(api_key="mock_key")
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = f"```json\n{json.dumps(mock_llm_json)}\n```"
    mock_client.models.generate_content.return_value = mock_resp

    with patch.object(analyzer, "_get_client", return_value=mock_client), \
         patch.object(analyzer, "is_gemini_available", return_value=True):
        res = analyzer.analyze(dialogue, use_gemini=True)
        assert len(res.key_highlights) == 1
        assert res.key_highlights[0].category == "Milestone"
        assert len(res.potential_tasks) == 1
        assert res.potential_tasks[0].urgency == "High"
        assert "Caching Subsystem" in res.topics_discussed

    # Test empty / no speech input
    res_empty = analyzer.analyze([], use_gemini=False)
    assert res_empty.summary == "No audible speech detected to analyze."
    assert res_empty.key_highlights == []
    assert res_empty.potential_tasks == []

    # Test ConversationSession input
    session = ConversationSession(
        session_id="session_test",
        power_axis="LATERAL",
        target_speaker="USER",
        dialogue=[Utterance(speaker="USER", start_time=0.0, end_time=2.0, transcript="Let's sync up later today.")]
    )
    res_session = analyzer.analyze(session, use_gemini=False)
    assert res_session.session_id == "transcription_analysis" or res_session.session_id == "session_test"
    assert len(res_session.potential_tasks) >= 1
