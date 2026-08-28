"""
Automated unit tests for LLM Executive Coaching Engine and Schemas.
"""

import pytest
from engine.schema import (
    Utterance,
    ConversationSession,
    ExecutiveCoachingEvaluation,
    CommunicationMetrics,
    TopStrength,
    AreaForImprovement,
    FillerWordMetric
)
from engine.persona_ontology import PowerAxis
from engine.coaching_engine import ExecutiveCoachingEngine


@pytest.fixture
def sample_session():
    dialogue = [
        Utterance(speaker="COUNTERPART", start_time=0.0, end_time=3.0, transcript="What is our status on the rollout?"),
        Utterance(speaker="USER", start_time=3.2, end_time=9.0, transcript="Basically, matlab I just think maybe we could ship next week."),
        Utterance(speaker="COUNTERPART", start_time=9.5, end_time=13.0, transcript="Can we commit to Thursday?"),
        Utterance(speaker="USER", start_time=13.2, end_time=20.0, transcript="Understood. Our data demonstrates that the build is stable and we have decided to ship on Thursday morning.")
    ]
    return ConversationSession(
        session_id="test_session_101",
        timestamp_utc="2026-08-27T18:00:00Z",
        target_speaker="USER",
        counterpart_name="VP of Product",
        counterpart_role="VP of Product",
        power_axis="UPWARD",
        dialogue=dialogue
    )


def test_schema_constraints_and_bounds():
    metrics = CommunicationMetrics(
        presence_score=85,
        assertiveness_score=90,
        active_listening_score=80,
        filler_words_detected=[FillerWordMetric(token="basically", count=2)]
    )
    assert 0 <= metrics.presence_score <= 100
    assert 0 <= metrics.assertiveness_score <= 100
    assert 0 <= metrics.active_listening_score <= 100

    strength = TopStrength(
        observation="Strong and decisive delivery.",
        verbatim_quote="Our data demonstrates that the build is stable."
    )
    assert len(strength.observation) <= 250

    improvement = AreaForImprovement(
        critique="Too much verbal hesitation.",
        verbatim_quote="Basically, matlab I just think",
        coached_phrasing="The rollout is scheduled for Thursday."
    )
    assert len(improvement.critique) <= 250
    assert len(improvement.coached_phrasing) <= 250


def test_coaching_engine_dynamic_and_capped_items(sample_session):
    engine = ExecutiveCoachingEngine()
    
    # Test dynamic / unconstrained count
    eval_dynamic = engine.evaluate_session(sample_session, top_n=None, use_llm=False)
    assert len(eval_dynamic.top_strengths) >= 1
    assert len(eval_dynamic.areas_for_improvement) >= 1
    assert 0 <= eval_dynamic.metrics.presence_score <= 100
    assert 0 <= eval_dynamic.metrics.assertiveness_score <= 100
    assert 0 <= eval_dynamic.metrics.active_listening_score <= 100

    # Test explicit cap top_n = 1
    eval_1 = engine.evaluate_session(sample_session, top_n=1, use_llm=False)
    assert len(eval_1.top_strengths) <= 1
    assert len(eval_1.areas_for_improvement) <= 1


def test_persona_upward_coaching_content(sample_session):
    engine = ExecutiveCoachingEngine()
    evaluation = engine.evaluate_session(sample_session, top_n=2, use_llm=False)
    
    assert "UPWARD" in evaluation.persona_context
    assert "BLUF" in evaluation.persona_alignment_notes or "Upward" in evaluation.persona_alignment_notes
    assert any(f.token == "matlab" for f in evaluation.metrics.filler_words_detected)


def test_phonetic_and_vocal_fillers_detection():
    from engine.metrics_calculator import MetricsCalculator

    test_speech = "Ummm, so basically I was thinking hmm, we need aaah to test aaaa our new service like right now."
    fillers = MetricsCalculator.detect_fillers(test_speech)
    
    tokens = {f.token for f in fillers}
    # Verify phonetic elongations are detected
    assert "ummm" in tokens
    assert "hmm" in tokens
    assert "aaah" in tokens
    assert "aaaa" in tokens
    # Verify English discourse fillers are detected
    assert "basically" in tokens
    assert "like" in tokens
