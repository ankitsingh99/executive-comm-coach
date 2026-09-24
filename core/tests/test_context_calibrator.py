"""
Unit tests for CommunicationContextCalibrator (Multimodal context gauging).
"""

from engine.context_calibrator import CommunicationContextCalibrator
from engine.persona_ontology import PowerAxis
from engine.schema import Utterance, AcousticAnalysisResult, SpeakerAcousticProfile


def test_infer_solo_context():
    utterances = [
        Utterance(
            speaker="USER",
            start_time=0.0,
            end_time=5.0,
            transcript="Today I am going to present the quarterly roadmap and our main objectives.",
        )
    ]
    acoustic_res = AcousticAnalysisResult(
        detected_speaker_count=1,
        is_multi_speaker=False,
        overall_tone="Calm & Measured",
        speakers=[SpeakerAcousticProfile(speaker_id="SPEAKER_01", talk_time_percentage=100.0)],
    )

    inference = CommunicationContextCalibrator.infer_context(acoustic_res, utterances)
    assert inference.recommended_axis == PowerAxis.SOLO
    assert inference.confidence_score >= 0.70
    assert "Solo" in inference.acoustic_rationale or "Single voice" in inference.acoustic_rationale


def test_infer_casual_social_context():
    utterances = [
        Utterance(speaker="USER", start_time=0.0, end_time=3.0, transcript="Arre bhai kya haal hai chalo coffee pe?"),
        Utterance(
            speaker="COUNTERPART",
            start_time=3.2,
            end_time=6.0,
            transcript="Haan bhai sahi hai weekend pe chill karte hain.",
        ),
    ]
    acoustic_res = AcousticAnalysisResult(
        detected_speaker_count=2,
        is_multi_speaker=True,
        overall_tone="Dynamic & Expressive",
        speakers=[
            SpeakerAcousticProfile(speaker_id="USER", talk_time_percentage=50.0),
            SpeakerAcousticProfile(speaker_id="COUNTERPART", talk_time_percentage=50.0),
        ],
    )

    inference = CommunicationContextCalibrator.infer_context(acoustic_res, utterances)
    assert inference.recommended_axis == PowerAxis.CASUAL
    assert "colloquial" in inference.semantic_rationale.lower() or "informal" in inference.semantic_rationale.lower()


def test_infer_conflict_negotiation_context():
    utterances = [
        Utterance(
            speaker="USER",
            start_time=0.0,
            end_time=3.5,
            transcript="Yeh unfairness hai HR mana kar raha hai half salary pe.",
        ),
        Utterance(
            speaker="COUNTERPART",
            start_time=3.0,
            end_time=7.0,
            transcript="Par wahi problem hai compensation aur budget issue hai.",
            is_overlapping=True,
            overlap_duration_sec=0.5,
        ),
    ]
    acoustic_res = AcousticAnalysisResult(
        detected_speaker_count=2,
        is_multi_speaker=True,
        overall_tone="Assertive & Decisive",
        overlapping_speech_events=2,
        overlap_duration_total_sec=1.8,
        speakers=[
            SpeakerAcousticProfile(speaker_id="USER", talk_time_percentage=45.0),
            SpeakerAcousticProfile(speaker_id="COUNTERPART", talk_time_percentage=55.0),
        ],
    )

    inference = CommunicationContextCalibrator.infer_context(acoustic_res, utterances)
    assert inference.recommended_axis == PowerAxis.CONFLICT
    assert "Negotiation" in inference.semantic_rationale or "friction" in inference.semantic_rationale.lower()


def test_infer_lateral_collaboration_context():
    utterances = [
        Utterance(
            speaker="USER", start_time=0.0, end_time=4.0, transcript="Let us sync on the API release and PR review."
        ),
        Utterance(
            speaker="COUNTERPART",
            start_time=4.5,
            end_time=8.0,
            transcript="I will deploy the sprint feature to staging before the milestone.",
        ),
    ]
    acoustic_res = AcousticAnalysisResult(
        detected_speaker_count=2,
        is_multi_speaker=True,
        overall_tone="Calm & Measured",
        speakers=[
            SpeakerAcousticProfile(speaker_id="USER", talk_time_percentage=48.0),
            SpeakerAcousticProfile(speaker_id="COUNTERPART", talk_time_percentage=52.0),
        ],
    )

    inference = CommunicationContextCalibrator.infer_context(acoustic_res, utterances)
    assert inference.recommended_axis == PowerAxis.LATERAL
