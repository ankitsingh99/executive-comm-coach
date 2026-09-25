"""
Unit tests for Transcription Analyzer Engine (Key Highlights & Potential Tasks).
"""

from datetime import datetime

from engine.action_item_extractor import ActionItemExtractor
from engine.schema import ConversationSession, Utterance
from engine.transcription_analyzer import TranscriptionAnalyzer


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
    from unittest.mock import MagicMock, patch

    dialogue = [
        Utterance(
            speaker="USER", start_time=0.0, end_time=4.0, transcript="We will deploy the caching subsystem on Friday."
        )
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
                "importance": "High",
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
                "urgency": "High",
            }
        ],
        "topics_discussed": ["Caching Subsystem", "Deployment Schedule"],
        "sentiment_tone": "Decisive & Collaborative",
    }

    analyzer = TranscriptionAnalyzer(api_key="mock_key")
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = f"```json\n{json.dumps(mock_llm_json)}\n```"
    mock_client.models.generate_content.return_value = mock_resp

    with (
        patch.object(analyzer, "_get_client", return_value=mock_client),
        patch.object(analyzer, "is_gemini_available", return_value=True),
    ):
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
        dialogue=[Utterance(speaker="USER", start_time=0.0, end_time=2.0, transcript="Let's sync up later today.")],
    )
    res_session = analyzer.analyze(session, use_gemini=False)
    assert res_session.session_id == "transcription_analysis" or res_session.session_id == "session_test"
    assert len(res_session.potential_tasks) >= 1


def test_transcription_analyzer_helper_branches():
    """Test sentiment tones, summary variants, headline truncations, and normalization."""
    analyzer = TranscriptionAnalyzer()

    # 1. Sentiment tones
    assert (
        analyzer._detect_sentiment_tone("We have a major blocker and risk in deployment.") == "Urgent & Issue-Focused"
    )
    assert analyzer._detect_sentiment_tone("This is great and we are perfectly aligned.") == "Positive & Collaborative"
    assert analyzer._detect_sentiment_tone("We will ship and deliver on Thursday.") == "Decisive & Action-Oriented"
    assert analyzer._detect_sentiment_tone("Hello, how is the weather today?") == "Calm & Constructive"

    # 2. Summary variations
    from engine.schema import ActionItem, KeyHighlight

    utts = [Utterance(speaker="USER", start_time=0.0, end_time=2.0, transcript="Test")]
    # Highlights only
    hl = [KeyHighlight(headline="H", takeaway="Core architecture.", speaker="USER", verbatim_quote="q")]
    s_hl = analyzer._generate_summary(utts, hl, [])
    assert "focused on core architecture" in s_hl.lower()

    # Tasks only (no highlights)
    task = [ActionItem(owner="USER", task="Review PR")]
    s_task = analyzer._generate_summary(utts, [], task)
    assert "established 1 concrete action item" in s_task.lower()

    # Neither
    s_none = analyzer._generate_summary(utts, [], [])
    assert "covered key topical points" in s_none.lower()

    # 3. Normalization from string with empty lines, raw lines, and invalid input
    raw_str = "\n  \nSPEAKER_01: First turn\nPlain turn without speaker tag\n"
    normalized = analyzer._normalize_to_utterances(raw_str)
    assert len(normalized) == 2
    assert normalized[0].speaker == "SPEAKER_01"
    assert normalized[1].speaker == "USER"

    # Invalid input type
    assert analyzer._normalize_to_utterances(12345) == []

    # 4. Long headline truncation
    long_text = "we have decided to completely rebuild the whole caching system from scratch with redis"
    hd = analyzer._generate_highlight_headline(long_text, "Decision")
    assert hd.endswith("...")

    # 5. Empty text string normalization
    assert analyzer._normalize_to_utterances("") == []
    assert analyzer._normalize_to_utterances("   ") == []


def test_transcription_analyzer_client_and_gemini_fallbacks():
    """Test client error handling, is_gemini_available, and fallback when Gemini throws exception."""
    from unittest.mock import MagicMock, patch

    analyzer = TranscriptionAnalyzer(api_key="test_key")

    # 1. _get_client throws exception
    with patch("google.genai.Client", side_effect=Exception("Genai import failure")):
        assert analyzer._get_client() is None
        assert analyzer.is_gemini_available() is False

    # 2. _analyze_with_gemini returns None on API failure, fallback to NLP
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("API 500 error")

    utts = [
        Utterance(speaker="USER", start_time=0.0, end_time=3.0, transcript="I will send the report by 5 PM tomorrow.")
    ]
    with (
        patch.object(analyzer, "_get_client", return_value=mock_client),
        patch.object(analyzer, "is_gemini_available", return_value=True),
    ):
        res = analyzer.analyze(utts, use_gemini=True)
        # Should gracefully fallback to deterministic NLP analysis
        assert res is not None
        assert len(res.potential_tasks) >= 1

    # 3. _analyze_with_gemini when LLM returns no tasks
    mock_response = MagicMock()
    mock_response.text = '{"summary": "Test", "key_highlights": [], "potential_tasks": [], "topics_discussed": []}'
    mock_client.models.generate_content.side_effect = None
    mock_client.models.generate_content.return_value = mock_response

    with (
        patch.object(analyzer, "_get_client", return_value=mock_client),
        patch.object(analyzer, "is_gemini_available", return_value=True),
    ):
        res_no_tasks = analyzer.analyze(utts, use_gemini=True)
        assert res_no_tasks is not None
        # Tasks should have been extracted via deterministic fallback
        assert len(res_no_tasks.potential_tasks) >= 1
