"""
Unit and integration tests for overlapping speech detection, cross-talk analysis,
and interruption metrics/coaching.
"""

import pytest
from engine.schema import Utterance, ConversationSession
from asr_diarization.diarizer import DiarizationEngine
from engine.metrics_calculator import MetricsCalculator
from engine.local_coaching_synthesizer import LocalCoachingSynthesizer


def test_no_overlapping_speech_sequential():
    """Verify that cleanly separated sequential turns produce 0 overlaps and 0 interruptions."""
    utts = [
        Utterance(speaker="USER", start_time=0.0, end_time=4.0, transcript="Let's align on the roadmap milestones."),
        Utterance(speaker="COUNTERPART", start_time=4.5, end_time=8.0, transcript="Sounds good, I agree with this approach.")
    ]
    processed, overlap_events, total_dur, interruptions = DiarizationEngine.compute_overlapping_speech(utts)
    assert overlap_events == 0
    assert total_dur == 0.0
    assert interruptions == 0
    assert not processed[0].is_overlapping
    assert not processed[1].is_overlapping


def test_overlapping_cross_talk_detection():
    """Verify that simultaneous speech intervals are detected and tagged with duration."""
    utts = [
        Utterance(speaker="USER", start_time=1.0, end_time=5.0, transcript="We have decided to ship the release on Thursday morning."),
        Utterance(speaker="COUNTERPART", start_time=3.5, end_time=7.0, transcript="Wait, are we sure the database migrations are finished?")
    ]
    processed, overlap_events, total_dur, interruptions = DiarizationEngine.compute_overlapping_speech(utts)
    assert overlap_events >= 1
    assert total_dur == pytest.approx(1.5, 0.1) # 5.0 - 3.5 = 1.5s overlap
    assert interruptions == 1
    assert processed[0].is_overlapping
    assert processed[1].is_overlapping
    assert processed[1].interrupted_speaker == "USER"


def test_same_speaker_overlap_not_counted_as_cross_talk():
    """Consecutive segments by the same speaker should not count as inter-speaker cross-talk."""
    utts = [
        Utterance(speaker="USER", start_time=0.0, end_time=2.5, transcript="Part one of my update."),
        Utterance(speaker="USER", start_time=2.0, end_time=4.0, transcript="Part two of my update.")
    ]
    processed, overlap_events, total_dur, interruptions = DiarizationEngine.compute_overlapping_speech(utts)
    assert overlap_events == 0
    assert total_dur == 0.0
    assert interruptions == 0


def test_metrics_calculator_interruption_tracking():
    """Verify that MetricsCalculator records interruption and overlap counts."""
    utts = [
        Utterance(speaker="COUNTERPART", start_time=0.0, end_time=4.0, transcript="I was thinking we should delay the beta release."),
        Utterance(speaker="USER", start_time=2.5, end_time=6.0, transcript="Bilkul nahi, our data demonstrates caching reduces latency by 35%.")
    ]
    processed, _, _, _ = DiarizationEngine.compute_overlapping_speech(utts)
    metrics = MetricsCalculator.analyze_dialogue(processed, target_speaker="USER")

    assert metrics.interruption_count == 1
    assert metrics.overlap_count >= 1
    assert metrics.active_listening_score <= 100


def test_coaching_feedback_generated_for_interruption():
    """Verify that LocalCoachingSynthesizer produces actionable feedback when user interrupts."""
    utts = [
        Utterance(speaker="COUNTERPART", start_time=0.0, end_time=4.0, transcript="We need to review the Q3 budget constraints."),
        Utterance(speaker="USER", start_time=2.0, end_time=5.0, transcript="Basically matlab we have decided to deploy immediately.")
    ]
    processed, _, _, _ = DiarizationEngine.compute_overlapping_speech(utts)
    session = ConversationSession(
        session_id="overlap_session_1",
        dialogue=processed,
        power_axis="UPWARD",
        target_speaker="USER"
    )
    coach = LocalCoachingSynthesizer()
    evaluation = coach.synthesize(session, try_local_ollama=False)

    # Check that an area for improvement specifically notes cross-talk / interruption
    critiques = [imp.critique for imp in evaluation.areas_for_improvement]
    assert any("cross-talk" in c.lower() or "interruption" in c.lower() or "premature" in c.lower() for c in critiques)


def test_format_dialogue_cli_with_overlap():
    """Verify that CLI formatter includes visual overlap indicators."""
    utts = [
        Utterance(speaker="USER", start_time=0.0, end_time=3.0, transcript="First point.", is_overlapping=True, overlap_duration_sec=1.0),
        Utterance(speaker="COUNTERPART", start_time=2.0, end_time=5.0, transcript="Second point.", is_overlapping=True, interrupted_speaker="USER")
    ]
    cli_str = DiarizationEngine.format_dialogue_cli(utts, counterpart_name="RAHUL")
    assert "⚡" in cli_str
    assert "INTERRUPTED USER" in cli_str
