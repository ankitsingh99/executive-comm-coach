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
        transcript="Hey, I will call you on 31 aug at 10 am to review the proposal."
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
        transcript="Understood. We have decided to ship the release branch on Thursday morning."
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
        transcript="Please send me the updated latency metrics by Friday EOD."
    )
    items = ActionItemExtractor.extract_from_utterance(utterance)
    assert len(items) == 1
    item = items[0]
    assert item.owner == "Priya"
    assert "Request" in item.category
    assert "friday" in item.due_time_or_date.lower()


def test_action_item_extraction_review_and_investigation():
    utterance = Utterance(
        speaker="USER",
        start_time=0.0,
        end_time=4.0,
        transcript="I will review the PR by tomorrow afternoon."
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
        transcript="It was a great discussion yesterday and the weather was really nice."
    )
    items = ActionItemExtractor.extract_from_utterance(utterance)
    assert len(items) == 0


def test_coaching_engine_populates_action_items():
    dialogue = [
        Utterance(speaker="Rahul", start_time=0.0, end_time=4.0, transcript="I will call you on 31 aug at 10 am."),
        Utterance(speaker="USER", start_time=4.5, end_time=9.0, transcript="We have decided to ship the release on Friday morning.")
    ]
    session = ConversationSession(
        session_id="test_act_123",
        timestamp_utc="2026-08-28T23:00:00Z",
        target_speaker="USER",
        counterpart_name="Rahul",
        counterpart_role="Peer",
        power_axis="LATERAL",
        dialogue=dialogue
    )
    engine = ExecutiveCoachingEngine(use_local_only=True)
    evaluation = engine.evaluate_session(session)

    assert len(evaluation.action_items) >= 2
    owners = [ai.owner for ai in evaluation.action_items]
    assert "Rahul" in owners
    assert "USER" in owners
