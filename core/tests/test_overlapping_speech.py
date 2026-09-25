"""
Unit and integration tests for overlapping speech detection, cross-talk analysis,
and interruption metrics/coaching.
"""

import pytest
from asr_diarization.diarizer import DiarizationEngine
from engine.local_coaching_synthesizer import LocalCoachingSynthesizer
from engine.metrics_calculator import MetricsCalculator
from engine.schema import ConversationSession, Utterance


def test_no_overlapping_speech_sequential():
    """Verify that cleanly separated sequential turns produce 0 overlaps and 0 interruptions."""
    utts = [
        Utterance(speaker="USER", start_time=0.0, end_time=4.0, transcript="Let's align on the roadmap milestones."),
        Utterance(
            speaker="COUNTERPART", start_time=4.5, end_time=8.0, transcript="Sounds good, I agree with this approach."
        ),
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
        Utterance(
            speaker="USER",
            start_time=1.0,
            end_time=5.0,
            transcript="We have decided to ship the release on Thursday morning.",
        ),
        Utterance(
            speaker="COUNTERPART",
            start_time=3.5,
            end_time=7.0,
            transcript="Wait, are we sure the database migrations are finished?",
        ),
    ]
    processed, overlap_events, total_dur, interruptions = DiarizationEngine.compute_overlapping_speech(utts)
    assert overlap_events >= 1
    assert total_dur == pytest.approx(1.5, 0.1)  # 5.0 - 3.5 = 1.5s overlap
    assert interruptions == 1
    assert processed[0].is_overlapping
    assert processed[1].is_overlapping
    assert processed[1].interrupted_speaker == "USER"


def test_same_speaker_overlap_not_counted_as_cross_talk():
    """Consecutive segments by the same speaker should not count as inter-speaker cross-talk."""
    utts = [
        Utterance(speaker="USER", start_time=0.0, end_time=2.5, transcript="Part one of my update."),
        Utterance(speaker="USER", start_time=2.0, end_time=4.0, transcript="Part two of my update."),
    ]
    processed, overlap_events, total_dur, interruptions = DiarizationEngine.compute_overlapping_speech(utts)
    assert overlap_events == 0
    assert total_dur == 0.0
    assert interruptions == 0


def test_metrics_calculator_interruption_tracking():
    """Verify that MetricsCalculator records interruption and overlap counts."""
    utts = [
        Utterance(
            speaker="COUNTERPART",
            start_time=0.0,
            end_time=4.0,
            transcript="I was thinking we should delay the beta release.",
        ),
        Utterance(
            speaker="USER",
            start_time=2.5,
            end_time=6.0,
            transcript="Bilkul nahi, our data demonstrates caching reduces latency by 35%.",
        ),
    ]
    processed, _, _, _ = DiarizationEngine.compute_overlapping_speech(utts)
    metrics = MetricsCalculator.analyze_dialogue(processed, target_speaker="USER")

    assert metrics.interruption_count == 1
    assert metrics.overlap_count >= 1
    assert metrics.active_listening_score <= 100


def test_coaching_feedback_generated_for_interruption():
    """Verify that LocalCoachingSynthesizer produces actionable feedback when user interrupts."""
    utts = [
        Utterance(
            speaker="COUNTERPART",
            start_time=0.0,
            end_time=4.0,
            transcript="We need to review the Q3 budget constraints.",
        ),
        Utterance(
            speaker="USER",
            start_time=2.0,
            end_time=5.0,
            transcript="Basically matlab we have decided to deploy immediately.",
        ),
    ]
    processed, _, _, _ = DiarizationEngine.compute_overlapping_speech(utts)
    session = ConversationSession(
        session_id="overlap_session_1", dialogue=processed, power_axis="UPWARD", target_speaker="USER"
    )
    coach = LocalCoachingSynthesizer()
    evaluation = coach.synthesize(session, try_local_ollama=False)

    # Check that an area for improvement specifically notes cross-talk / interruption
    critiques = [imp.critique for imp in evaluation.areas_for_improvement]
    assert any("cross-talk" in c.lower() or "interruption" in c.lower() or "premature" in c.lower() for c in critiques)


def test_format_dialogue_cli_with_overlap():
    """Verify that CLI formatter includes visual overlap indicators."""
    utts = [
        Utterance(
            speaker="USER",
            start_time=0.0,
            end_time=3.0,
            transcript="First point.",
            is_overlapping=True,
            overlap_duration_sec=1.0,
        ),
        Utterance(
            speaker="COUNTERPART",
            start_time=2.0,
            end_time=5.0,
            transcript="Second point.",
            is_overlapping=True,
            interrupted_speaker="USER",
        ),
    ]
    cli_str = DiarizationEngine.format_dialogue_cli(utts, counterpart_name="RAHUL")
    assert "⚡" in cli_str
    assert "INTERRUPTED USER" in cli_str


def test_n_speaker_conversation_analysis():
    """
    Validate that conversations with N >= 4 participants (e.g. USER, RAHUL, PRIYA, SANDEEP)
    are accurately analyzed for multi-party speech-to-text, overlaps, individual speaker turns,
    action items, emotional trajectories, and consensus agreements.
    """
    from engine.action_item_extractor import ActionItemExtractor
    from engine.conversational_intelligence_engine import ConversationalIntelligenceEngine

    dialogue = [
        Utterance(
            speaker="USER", start_time=0.0, end_time=4.0, transcript="Welcome team, let's review the Q3 launch plan."
        ),
        Utterance(
            speaker="RAHUL",
            start_time=3.5,
            end_time=7.0,
            transcript="Hey I am Rahul, I will deploy the database migrations on Thursday at 10 am.",
        ),
        Utterance(
            speaker="PRIYA",
            start_time=6.8,
            end_time=10.0,
            transcript="Priya here. Wait, I have an open concern regarding frontend latency regression.",
        ),
        Utterance(
            speaker="SANDEEP",
            start_time=9.5,
            end_time=13.0,
            transcript="Sandeep here. We agree to run load tests before Thursday's deployment.",
        ),
        Utterance(
            speaker="USER", start_time=13.5, end_time=16.0, transcript="Perfect, agreed on running load tests first."
        ),
    ]

    # 1. Multi-party overlap and cross-talk computation
    processed, overlap_events, total_dur, interruptions = DiarizationEngine.compute_overlapping_speech(dialogue)
    assert overlap_events == 3  # (USER, RAHUL), (RAHUL, PRIYA), (PRIYA, SANDEEP)
    assert total_dur > 0.5
    assert interruptions == 3

    # Check distinct speakers
    distinct_speakers = {u.speaker for u in processed}
    assert distinct_speakers == {"USER", "RAHUL", "PRIYA", "SANDEEP"}

    # 2. Action item extraction across multiple owners
    actions = ActionItemExtractor.extract_from_dialogue(processed)
    assert len(actions) >= 1
    # Check that Rahul's commitment was extracted with correct owner
    rahul_action = next((a for a in actions if a.owner.upper() == "RAHUL"), None)
    assert rahul_action is not None
    assert "Thursday" in (rahul_action.due_time_or_date or "")

    # 3. Conversational dynamics & emotional trajectory across all N participants
    dyn, emo, agr, loops = ConversationalIntelligenceEngine.analyze_session(processed, target_speaker="USER")
    assert len(emo) == 5
    assert {e.speaker for e in emo} == {"USER", "RAHUL", "PRIYA", "SANDEEP"}

    # 4. Consensus agreements & open loops
    assert len(agr) >= 1
    assert any("load test" in a.agreed_solution.lower() or "agree" in a.headline.lower() for a in agr)

    # 5. Open loop raised by Priya
    assert len(loops) >= 1
    priya_loop = next((loop_item for loop_item in loops if "PRIYA" in loop_item.raised_by.upper()), None)
    assert priya_loop is not None
    assert "latency" in priya_loop.concern_topic.lower() or "latency" in priya_loop.context.lower()

    # 6. Test format_dialogue_markdown and CLI formatting with user_name
    md = DiarizationEngine.format_dialogue_markdown(processed)
    assert "**USER**" in md
    assert "**RAHUL**" in md
    assert "Interrupted" in md or "Overlapping" in md

    cli_solo = DiarizationEngine.format_dialogue_cli(
        [Utterance(speaker="USER", start_time=0.0, end_time=3.0, transcript="Solo rehearsal.")], user_name="Ashish"
    )
    assert "[ASHISH (Solo)]" in cli_solo or "[ASHISH / YOU]" in cli_solo

    # 7. Test assign_roles with recognized user & counterpart names
    raw_utts = [
        Utterance(speaker="SPEAKER_01", start_time=0.0, end_time=2.0, transcript="Hello"),
        Utterance(speaker="SPEAKER_02", start_time=2.0, end_time=4.0, transcript="Hi there"),
        Utterance(speaker="CUSTOM_SPK", start_time=4.0, end_time=6.0, transcript="Testing"),
    ]
    assigned = DiarizationEngine.assign_roles(
        raw_utts, user_speaker_id="SPEAKER_01", recognized_user_name="Ashish", recognized_counterpart_name="Vikram"
    )
    assert assigned[0].speaker == "Ashish"
    assert assigned[1].speaker == "Vikram"
    assert assigned[2].speaker == "CUSTOM_SPK"

    # 8. Test detect_and_apply_verbal_introductions re-tagging
    intro_utts = [
        Utterance(speaker="USER", start_time=0.0, end_time=3.0, transcript="Hi, I am Ashish from engineering."),
        Utterance(speaker="COUNTERPART", start_time=3.5, end_time=7.0, transcript="Hey, this is Vikram from product."),
    ]
    retagged, cp, usr = DiarizationEngine.detect_and_apply_verbal_introductions(intro_utts)
    assert usr == "Ashish"
    assert cp == "Vikram"
    assert retagged[0].speaker == "Ashish"
    assert retagged[1].speaker == "Vikram"

    # 9. Format CLI with invalid / zero end time
    zero_dur_utt = [Utterance(speaker="USER", start_time=5.0, end_time=5.0, transcript="Quick check.")]
    cli_zero = DiarizationEngine.format_dialogue_cli(zero_dur_utt)
    assert "Quick check." in cli_zero
