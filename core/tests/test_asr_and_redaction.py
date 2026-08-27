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
