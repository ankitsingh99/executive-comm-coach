"""
Automated tests for ASR, Diarization, VAD Acoustic Gating, and Privacy Redaction.
"""

from asr_diarization.vad_gater import AmbientVadGate
from asr_diarization.diarizer import DiarizationEngine
from asr_diarization.sarvam_client import SarvamSpeechClient
from privacy.pii_redactor import PIIRedactor
from engine.schema import Utterance


def test_vad_acoustic_gater_trigger():
    vad = AmbientVadGate(speech_prob_threshold=0.75, sustained_window_ms=600.0)
    
    # Inactive frame should not trigger
    triggered, msg = vad.evaluate_frame(timestamp_ms=0, speech_prob=0.2)
    assert not triggered
    assert "Acoustic gate" in msg

    # Sustained frames above 0.75 across window should trigger
    for t in range(100, 900, 100):
        triggered, msg = vad.evaluate_frame(timestamp_ms=t, speech_prob=0.85)

    assert triggered
    assert "Sustained speech detected" in msg


def test_diarization_engine_role_assignment():
    raw = [
        Utterance(speaker="SPEAKER_01", start_time=0.0, end_time=2.0, transcript="Turn 1"),
        Utterance(speaker="SPEAKER_02", start_time=2.1, end_time=4.0, transcript="Turn 2")
    ]
    aligned = DiarizationEngine.assign_roles(raw, user_speaker_id="SPEAKER_01")
    assert aligned[0].speaker == "USER"
    assert aligned[1].speaker == "COUNTERPART"

    cli_formatted = DiarizationEngine.format_dialogue_cli(aligned)
    assert "[USER / YOU]" in cli_formatted
    assert "[COUNTERPART]" in cli_formatted


def test_verbal_self_introduction_extraction():
    # Test individual phrase extractions
    assert DiarizationEngine.extract_speaker_name_from_text("hey i am rahul and today we will discuss the project") == "Rahul"
    assert DiarizationEngine.extract_speaker_name_from_text("Vikram here. Can you give me an update?") == "Vikram"
    assert DiarizationEngine.extract_speaker_name_from_text("Hi, this is Priya Sharma from product") == "Priya Sharma"
    assert DiarizationEngine.extract_speaker_name_from_text("I am thinking we should finish by Friday") is None

    # Test dialogue auto-tagging (Counterpart introduction)
    dialogue = [
        Utterance(speaker="COUNTERPART", start_time=0.0, end_time=3.0, transcript="Hey I am Rahul and today I want to sync on our goals."),
        Utterance(speaker="USER", start_time=3.2, end_time=6.0, transcript="Hi Rahul, sounds great."),
        Utterance(speaker="COUNTERPART", start_time=6.2, end_time=9.0, transcript="Let us begin with the timeline.")
    ]
    updated, counterpart_name, user_name = DiarizationEngine.detect_and_apply_verbal_introductions(dialogue, user_speaker_id="USER")
    assert counterpart_name == "Rahul"
    assert updated[0].speaker == "Rahul"
    assert updated[1].speaker == "USER"
    assert updated[2].speaker == "Rahul"

    # Test solo user monologue auto-tagging
    solo_dialogue = [
        Utterance(speaker="USER", start_time=0.0, end_time=5.0, transcript="Hey I am Ashish and today I will present the architecture.")
    ]
    updated_solo, c_name, u_name = DiarizationEngine.detect_and_apply_verbal_introductions(solo_dialogue, user_speaker_id="USER")
    assert u_name == "Ashish"
    assert updated_solo[0].speaker == "Ashish"

    solo_cli = DiarizationEngine.format_dialogue_cli(updated_solo, user_name="Ashish")
    assert "[ASHISH (Solo)]" in solo_cli


def test_sarvam_client_diarization_parsing():
    client = SarvamSpeechClient(api_key="mock_key")
    mock_payload = {
        "transcript": "Hello Vikram, here is the update.",
        "diarized_transcript": {
            "entries": [
                {
                    "speaker_id": "speaker_0",
                    "start_time_seconds": 0.0,
                    "end_time_seconds": 2.5,
                    "transcript": "Hello Vikram, here is the status."
                },
                {
                    "speaker_id": "speaker_1",
                    "start_time_seconds": 2.6,
                    "end_time_seconds": 5.0,
                    "transcript": "Thanks, what is the latency?"
                }
            ]
        }
    }
    utterances = client._parse_sarvam_response(mock_payload)
    assert len(utterances) == 2
    assert utterances[0].speaker == "USER"
    assert utterances[0].transcript == "Hello Vikram, here is the status."
    assert utterances[1].speaker == "COUNTERPART"
    assert utterances[1].transcript == "Thanks, what is the latency?"


def test_pii_redactor_all_categories():
    raw_text = (
        "Call me at +919876543210 or email me at priya@corp.internal. "
        "My PAN is ABCDE1234F, Aadhaar is 1234 5678 9012. "
        "The project budget is ₹25 lakh and the API token secret: my_secret_token_123."
    )
    redacted, counts = PIIRedactor.redact_text(raw_text)

    assert "[REDACTED_PHONE]" in redacted
    assert "[REDACTED_EMAIL]" in redacted
    assert "[REDACTED_PAN]" in redacted
    assert "[REDACTED_AADHAAR]" in redacted
    assert "[REDACTED_FINANCIAL]" in redacted
    assert "[REDACTED_SECRET]" in redacted
    assert counts["PHONE"] == 1
    assert counts["EMAIL"] == 1
    assert counts["PAN"] == 1
    assert counts["AADHAAR"] == 1
    assert counts["FINANCIAL"] == 1
    assert counts["CREDENTIAL"] == 1
