"""
Unit tests for Action Item, Commitment & Follow-up Extraction Engine.
"""

from engine.schema import Utterance, ConversationSession, ExecutiveCoachingEvaluation
from engine.action_item_extractor import ActionItemExtractor
from engine.coaching_engine import ExecutiveCoachingEngine


def test_action_item_extraction_scheduling_and_calls():
    utterance = Utterance(
        speaker="Rahul",
        start_time=0.0,
        end_time=4.0,
        transcript="Hey, I will call you on 31 aug at 10 am to review the proposal.",
    )
    items = ActionItemExtractor.extract_from_utterance(utterance)
    assert len(items) == 1
    item = items[0]
    assert item.owner == "Rahul"
    assert "Call" in item.category
    assert "31 aug at 10 am" in item.due_time_or_date.lower()
    assert "31 aug at 10 am" in item.verbatim_quote.lower()


def test_action_item_extraction_deliverable_and_deadline():
    utterance = Utterance(
        speaker="USER",
        start_time=0.0,
        end_time=5.0,
        transcript="Understood. We have decided to ship the release branch on Thursday morning.",
    )
    items = ActionItemExtractor.extract_from_utterance(utterance)
    assert len(items) == 1
    item = items[0]
    assert item.owner == "USER"
    assert "Deliverable" in item.category
    assert "thursday" in item.due_time_or_date.lower()


def test_action_item_extraction_delegation_and_requests():
    utterance = Utterance(
        speaker="Priya",
        start_time=0.0,
        end_time=4.0,
        transcript="Please send me the updated latency metrics by Friday EOD.",
    )
    items = ActionItemExtractor.extract_from_utterance(utterance)
    assert len(items) == 1
    item = items[0]
    assert item.owner == "Priya"
    assert "Request" in item.category
    assert "friday" in item.due_time_or_date.lower()


def test_action_item_extraction_review_and_investigation():
    utterance = Utterance(
        speaker="USER", start_time=0.0, end_time=4.0, transcript="I will review the PR by tomorrow afternoon."
    )
    items = ActionItemExtractor.extract_from_utterance(utterance)
    assert len(items) == 1
    item = items[0]
    assert item.owner == "USER"
    assert "Review" in item.category
    assert "tomorrow" in item.due_time_or_date.lower()
    assert item.urgency == "High"


def test_action_item_extraction_casual_dialogue_negative():
    utterance = Utterance(
        speaker="USER",
        start_time=0.0,
        end_time=3.0,
        transcript="It was a great discussion yesterday and the weather was really nice.",
    )
    items = ActionItemExtractor.extract_from_utterance(utterance)
    assert len(items) == 0


def test_coaching_engine_populates_action_items():
    dialogue = [
        Utterance(speaker="Rahul", start_time=0.0, end_time=4.0, transcript="I will call you on 31 aug at 10 am."),
        Utterance(
            speaker="USER",
            start_time=4.5,
            end_time=9.0,
            transcript="We have decided to ship the release on Friday morning.",
        ),
    ]
    session = ConversationSession(
        session_id="test_act_123",
        timestamp_utc="2026-08-28T23:00:00Z",
        target_speaker="USER",
        counterpart_name="Rahul",
        counterpart_role="Peer",
        power_axis="LATERAL",
        dialogue=dialogue,
    )
    engine = ExecutiveCoachingEngine(use_local_only=True)
    evaluation = engine.evaluate_session(session)

    assert len(evaluation.action_items) >= 2
    assert hasattr(evaluation, "key_highlights")
    assert len(evaluation.key_highlights) >= 1
    owners = [ai.owner for ai in evaluation.action_items]
    assert "Rahul" in owners
    assert "USER" in owners


def test_action_item_ambiguous_time_resolution():
    from datetime import datetime

    ref_dt = datetime(2026, 9, 12, 11, 0, 0)
    utterance = Utterance(speaker="Ankit", start_time=0.0, end_time=3.0, transcript="I will call Rahul at 9.")
    items = ActionItemExtractor.extract_from_utterance(utterance, ref_dt=ref_dt)
    assert len(items) == 1
    assert items[0].target_time_inferred_ampm == "PM"
    assert "2026-09-12T21:00:00" in items[0].resolved_datetime


def test_action_item_edge_cases_and_deduplication():
    """Test short utterances, long descriptions, fallback anchors, and deduplication."""
    # 1. Short utterance (< 8 chars)
    short_u = Utterance(speaker="USER", start_time=0.0, end_time=1.0, transcript="ok")
    assert ActionItemExtractor.extract_from_utterance(short_u) == []

    # 2. Long task description (> 120 chars)
    desc = ActionItemExtractor.extract_task_description(
        "I will definitely deliver our entire distributed database architecture across all global regions and environments by next quarter with zero downtime.",
        matched_intent="Deliverable",
    )
    assert desc.endswith("...")

    # 3. Deduplication of identical action items
    duplicate_dialogue = [
        Utterance(speaker="USER", start_time=0.0, end_time=2.0, transcript="I will send the report by 5 PM."),
        Utterance(speaker="USER", start_time=3.0, end_time=5.0, transcript="I will send the report by 5 PM.")
    ]
    deduped = ActionItemExtractor.extract_from_dialogue(duplicate_dialogue)
    assert len(deduped) == 1

    # 4. Fallback temporal anchor regex
    anchor = ActionItemExtractor.extract_temporal_anchor("The deadline is on 15 Oct 2026.")
    assert anchor is not None


def test_action_item_extractor_fallback_regex_and_unsplit_sentence():
    """Test extract_temporal_anchor fallback regex loop when TemporalResolver returns None, and unsplit sentences."""
    from unittest.mock import patch

    # 1. Temporal anchor regex fallback when TemporalResolver is mocked to return None
    with patch("engine.action_item_extractor.TemporalResolver.resolve_time_expression", return_value=None):
        anchor = ActionItemExtractor.extract_temporal_anchor("Please complete this by tomorrow at 5 pm.")
        assert anchor is not None
        assert "tomorrow" in anchor or "5 pm" in anchor

        no_anchor = ActionItemExtractor.extract_temporal_anchor("Hello there, nice to meet you.")
        assert no_anchor is None

    # 2. Utterance with no punctuation delimiters (exercises sentences = [text] fallback)
    u_no_punc = Utterance(speaker="USER", start_time=0.0, end_time=2.0, transcript="i will definitely send the logs tonight")
    items = ActionItemExtractor.extract_from_utterance(u_no_punc)
    assert len(items) == 1
    assert "send the logs" in items[0].task.lower()

