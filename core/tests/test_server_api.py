"""
Unit tests for Emulator Server API Endpoints including Transcription Analysis.
"""

import json
from datetime import datetime
from engine.transcription_analyzer import TranscriptionAnalyzer
from server import EmulatorHandler
from unittest.mock import MagicMock


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
        use_gemini=False
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
                "quote": kh.verbatim_quote
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
                "urgency": t.urgency
            }
            for t in res.potential_tasks
        ]
    }
    
    assert "summary" in dict_repr
    assert len(dict_repr["highlights"]) >= 1
    assert len(dict_repr["tasks"]) >= 1
    assert dict_repr["tasks"][0]["resolved_datetime"] is not None
