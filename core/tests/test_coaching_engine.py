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


def test_coaching_engine_exact_top_n(sample_session):
    engine = ExecutiveCoachingEngine()
    
    # Test N = 2
    eval_2 = engine.evaluate_session(sample_session, top_n=2, use_llm=False)
    assert len(eval_2.top_strengths) == 2
    assert len(eval_2.areas_for_improvement) == 2
    assert 0 <= eval_2.metrics.presence_score <= 100
    assert 0 <= eval_2.metrics.assertiveness_score <= 100
    assert 0 <= eval_2.metrics.active_listening_score <= 100

    # Test N = 3
    eval_3 = engine.evaluate_session(sample_session, top_n=3, use_llm=False)
    assert len(eval_3.top_strengths) == 3
    assert len(eval_3.areas_for_improvement) == 3


def test_persona_upward_coaching_content(sample_session):
    engine = ExecutiveCoachingEngine()
    evaluation = engine.evaluate_session(sample_session, top_n=2, use_llm=False)
    
    assert "UPWARD" in evaluation.persona_context
    assert "BLUF" in evaluation.persona_alignment_notes or "Upward" in evaluation.persona_alignment_notes
    assert any(f.token == "matlab" for f in evaluation.metrics.filler_words_detected)
