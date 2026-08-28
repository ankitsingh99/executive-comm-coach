"""
Unit tests for Multilingual & Hinglish Coaching, Takeaways, and Feedback.
"""

from engine.schema import Utterance, ConversationSession
from engine.coaching_engine import ExecutiveCoachingEngine
from engine.metrics_calculator import MetricsCalculator
from engine.action_item_extractor import ActionItemExtractor


def test_hinglish_fillers_and_hedging_detection():
    text = "Dekho basically matlab mujhe lagta hai ki hume shayad deployment delay karna padega."
    fillers = MetricsCalculator.detect_fillers(text)
    filler_tokens = [f.token for f in fillers]
    assert "matlab" in filler_tokens or "basically" in filler_tokens

    hedging_count, assertive_count = MetricsCalculator.calculate_hedging_vs_assertion(text)
    assert hedging_count >= 1  # 'mujhe lagta hai' or 'shayad'


def test_hinglish_action_item_extraction():
    dialogue = [
        Utterance(speaker="Rahul", start_time=0.0, end_time=4.0, transcript="Main kal 10 baje call karunga aur metrics discuss karenge."),
        Utterance(speaker="USER", start_time=4.5, end_time=9.0, transcript="Theek hai, hum kal shaam tak release ship kar denge.")
    ]
    items = ActionItemExtractor.extract_from_dialogue(dialogue)
    assert len(items) >= 2
    
    rahul_item = next(item for item in items if item.owner == "Rahul")
    assert "call" in rahul_item.category.lower() or "meeting" in rahul_item.category.lower()
    assert "kal" in rahul_item.due_time_or_date.lower() or "10 baje" in rahul_item.due_time_or_date.lower()


def test_hinglish_coaching_takeaways_and_rephrasing():
    dialogue = [
        Utterance(speaker="Sandeep", start_time=0.0, end_time=3.0, transcript="Production latency ka status kya hai?"),
        Utterance(speaker="USER", start_time=3.5, end_time=8.0, transcript="Matlab mujhe lagta hai ki hume caching enable karni chahiye, par thoda doubt hai.")
    ]
    session = ConversationSession(
        session_id="test_hinglish_123",
        timestamp_utc="2026-08-28T23:00:00Z",
        target_speaker="USER",
        counterpart_name="Sandeep",
        counterpart_role="VP of Engineering",
        power_axis="UPWARD",
        dialogue=dialogue
    )

    engine = ExecutiveCoachingEngine(use_local_only=True)
    evaluation = engine.evaluate_session(session)

    # Verify takeaways are non-empty and relevant
    assert len(evaluation.longitudinal_summary) > 20
    assert "Action:" in evaluation.longitudinal_summary or "recommendation" in evaluation.longitudinal_summary.lower()

    # Verify critique and coached phrasing exist
    assert len(evaluation.areas_for_improvement) >= 1
    first_imp = evaluation.areas_for_improvement[0]
    assert len(first_imp.critique) > 10
    assert len(first_imp.coached_phrasing) > 10
    assert "Action:" in first_imp.critique
