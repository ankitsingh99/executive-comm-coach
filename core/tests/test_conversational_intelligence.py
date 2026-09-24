"""
Unit tests for Conversational Intelligence & Dynamics Engine.
Validates talk-time parity, turn latency, ask vs tell ratios, brevity index,
emotional trajectory, consensus agreement extraction, and open loops detection.
"""

import pytest
from core.engine.schema import (
    Utterance,
    ConversationSession,
    AcousticAnalysisResult,
    ConversationalDynamicsMetric,
    EmotionalTrajectoryPoint,
    AgreementPoint,
    UnresolvedOpenLoop,
)
from core.engine.conversational_intelligence_engine import ConversationalIntelligenceEngine
from core.engine.local_coaching_synthesizer import LocalCoachingSynthesizer


def test_conversational_dynamics_talk_time_and_latency():
    turns = [
        Utterance(
            speaker="USER", start_time=0.0, end_time=4.0, transcript="What is the latest status on the migration?"
        ),
        Utterance(
            speaker="PRIYA",
            start_time=4.5,
            end_time=9.5,
            transcript="We completed the initial schema migration yesterday.",
        ),
        Utterance(
            speaker="USER", start_time=10.0, end_time=13.0, transcript="Building on what you said, let's lock this in."
        ),
    ]

    dynamics, trajectory, agreements, open_loops = ConversationalIntelligenceEngine.analyze_session(
        turns, target_speaker="USER"
    )

    assert isinstance(dynamics, ConversationalDynamicsMetric)
    assert dynamics.user_words_total > 0
    assert dynamics.counterpart_words_total > 0
    # Latency: (4.5 - 4.0 = 0.5s -> 500ms), (10.0 - 9.5 = 0.5s -> 500ms) -> avg 500ms
    assert 400.0 <= dynamics.average_turn_latency_ms <= 600.0
    assert dynamics.user_talk_time_pct > 0
    assert dynamics.counterpart_talk_time_pct > 0
    # USER has 1 question and 1 directive
    assert dynamics.inquiry_count >= 1
    assert dynamics.directive_count >= 1
    assert dynamics.ask_vs_tell_ratio > 0


def test_emotional_trajectory_and_tension_arc():
    turns = [
        Utterance(
            speaker="USER", start_time=0.0, end_time=3.0, transcript="I am excited about this launch! It looks great."
        ),
        Utterance(
            speaker="PRIYA",
            start_time=3.5,
            end_time=7.0,
            transcript="There is a major blocker and problem with the API latency.",
        ),
        Utterance(
            speaker="USER",
            start_time=7.5,
            end_time=10.0,
            transcript="We agree on the fallback plan, sounds like a deal.",
        ),
    ]

    trajectory = ConversationalIntelligenceEngine.extract_emotional_trajectory(turns)
    assert len(trajectory) == 3

    # Turn 1 should be positive
    assert trajectory[0].valence_score > 0.0
    assert "Enthusiastic" in trajectory[0].emotion_label or "Aligned" in trajectory[0].emotion_label

    # Turn 2 should be negative / concerned
    assert trajectory[1].valence_score < 0.0
    assert trajectory[1].tension_level == "HIGH"

    # Turn 3 should be constructive / agreement
    assert trajectory[2].valence_score > 0.0


def test_agreement_and_consensus_extraction():
    turns = [
        Utterance(
            speaker="RAHUL", start_time=0.0, end_time=4.0, transcript="Should we ship the new v2 endpoint tomorrow?"
        ),
        Utterance(
            speaker="USER",
            start_time=4.5,
            end_time=8.0,
            transcript="Yes, we agree on shipping v2 tomorrow, let's lock this in.",
        ),
        Utterance(speaker="RAHUL", start_time=8.5, end_time=12.0, transcript="Haan bilkul yahi karenge, done deal."),
    ]

    agreements = ConversationalIntelligenceEngine.extract_agreements(turns)
    assert len(agreements) >= 2
    assert any("lock this in" in a.verbatim_quote.lower() for a in agreements)
    assert any("done deal" in a.verbatim_quote.lower() for a in agreements)


def test_unresolved_open_loops_extraction():
    turns = [
        Utterance(
            speaker="PRIYA",
            start_time=0.0,
            end_time=4.0,
            transcript="We still need to figure out the database backup retention policy.",
        ),
        Utterance(
            speaker="USER", start_time=4.5, end_time=8.0, transcript="Let's table this for now, circle back on Monday."
        ),
    ]

    open_loops = ConversationalIntelligenceEngine.extract_open_loops(turns)
    assert len(open_loops) >= 2
    assert any("figure out" in ol.context.lower() for ol in open_loops)
    assert any("circle back" in ol.context.lower() or "table this" in ol.context.lower() for ol in open_loops)
    for ol in open_loops:
        assert ol.recommended_followup != ""


def test_local_coaching_synthesizer_attaches_all_insights():
    turns = [
        Utterance(
            speaker="USER",
            start_time=0.0,
            end_time=4.0,
            transcript="So what you're saying is we should test the model? Basically, let's do that.",
        ),
        Utterance(
            speaker="ROHAN", start_time=4.2, end_time=7.0, transcript="Agreed! We still need to figure out caching."
        ),
    ]

    session = ConversationSession(
        session_id="test_conv_intel",
        timestamp_utc="2026-09-24T12:00:00Z",
        target_speaker="USER",
        counterpart_name="Rohan",
        counterpart_role="Tech Lead",
        power_axis="LATERAL",
        dialogue=turns,
    )

    synth = LocalCoachingSynthesizer()
    evaluation = synth.synthesize(session, try_local_ollama=False)

    assert evaluation.dynamics is not None
    assert evaluation.dynamics.user_words_total > 0
    assert len(evaluation.emotional_trajectory) == 2
    assert len(evaluation.agreements) >= 1
    assert len(evaluation.unresolved_loops) >= 1
