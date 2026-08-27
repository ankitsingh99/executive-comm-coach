"""
Executive Communication Coach - End-to-End Core MVP Pipeline Demo.
Demonstrates:
  1. Ambient Acoustic VAD Gating
  2. DPDP Statutory Consent & Audible Chime
  3. Bilingual Hinglish STT & Speaker Diarization
  4. Privacy Redaction
  5. Persona Ontology & Relational Calibration (Upward, Lateral, Downward)
  6. Structured Executive Coaching Synthesis
  7. DPDP Statutory Erasure Wipe
"""

import sys
import os
import json
from datetime import datetime, timezone

# Ensure package is resolvable
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

from engine import (
    ExecutiveCoachingEngine,
    ConversationSession,
    Utterance,
    PowerAxis
)
from asr_diarization import (
    AmbientVadGate,
    SarvamSpeechClient,
    DiarizationEngine
)
from privacy import (
    PIIRedactor,
    DPDPComplianceManager
)


def run_pipeline_demo():
    print("=" * 80)
    print(" [EXEC COACH] CONVERSATIONAL INTELLIGENCE & COACHING PIPELINE")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Stage 1: Ambient VAD Gating Simulation
    # -------------------------------------------------------------------------
    print("\n[STAGE 1] Ambient Acoustic Gating (Silero VAD, 16kHz PCM)...")
    vad = AmbientVadGate(speech_prob_threshold=0.75, sustained_window_ms=600.0)
    
    # Simulate passive non-speech noise frames
    for t_ms in [0, 100, 200]:
        triggered, msg = vad.evaluate_frame(t_ms, speech_prob=0.15)
        print(f"  Frame at {t_ms}ms: tau=0.15 -> {msg}")

    # Simulate sustained human speech onset
    print("\n  >> Human dialogue begins...")
    triggered = False
    for t_ms in [300, 400, 500, 600, 700, 800, 900]:
        triggered, msg = vad.evaluate_frame(t_ms, speech_prob=0.88)
        if triggered:
            print(f"  Frame at {t_ms}ms: tau=0.88 -> [TRIGGER] {msg}")
            break

    # -------------------------------------------------------------------------
    # Stage 2: DPDP Statutory Consent & Audible Notification
    # -------------------------------------------------------------------------
    print("\n[STAGE 2] DPDP Act 2023 Compliance & Consent Initiation...")
    compliance_mgr = DPDPComplianceManager(storage_root="/tmp/exec_coach_storage")
    session_id = f"session_{int(datetime.now(timezone.utc).timestamp())}"
    
    chime_msg = compliance_mgr.trigger_audible_chime()
    print(f"  [CHIME] {chime_msg}")
    
    consent_record = compliance_mgr.log_session_consent(session_id, counterpart_notified=True)
    print(f"  [CONSENT] Statutory Consent Logged: Session ID={consent_record.session_id}, AES-256 Storage=Active")

    # -------------------------------------------------------------------------
    # Stage 3: Bilingual Hinglish ASR & Speaker Diarization
    # -------------------------------------------------------------------------
    print("\n[STAGE 3] Bilingual Hinglish STT & Speaker Diarization (Sarvam Saaras v3)...")
    sarvam_client = SarvamSpeechClient()
    raw_utterances = sarvam_client.transcribe_audio_chunk(audio_file_path="mock_audio.opus")
    aligned_dialogue = DiarizationEngine.assign_roles(raw_utterances, user_speaker_id="USER")

    print("\n  --- Captured Diarized Hinglish Dialogue ---")
    formatted_md = DiarizationEngine.format_dialogue_markdown(aligned_dialogue)
    for line in formatted_md.split("\n\n"):
        print(f"  {line}")

    # -------------------------------------------------------------------------
    # Stage 4: Local Privacy-Preserving PII Redaction
    # -------------------------------------------------------------------------
    print("\n[STAGE 4] Privacy Redaction (Local Scrubber)...")
    sample_sensitive_turn = "Hey Reshma, send the API token secret: tok_83921048 to my email ashish@enterprise.internal or call +919876543210 regarding our Rs. 45 lakh budget."
    redacted_sample, red_counts = PIIRedactor.redact_text(sample_sensitive_turn)
    print(f"  Raw:      \"{sample_sensitive_turn}\"")
    print(f"  Redacted: \"{redacted_sample}\"")
    print(f"  Redaction Metrics: {red_counts}")

    # -------------------------------------------------------------------------
    # Stage 5: Persona Context Injection & Executive Coaching Synthesis
    # -------------------------------------------------------------------------
    print("\n[STAGE 5] LLM Executive Coaching Synthesis across Personas...")
    engine = ExecutiveCoachingEngine()

    personas_to_test = [
        ("Sandeep Sharma", "VP of Engineering", PowerAxis.UPWARD.value),
        ("Pooja Nair", "Principal Product Manager", PowerAxis.LATERAL.value),
        ("Kunal Verma", "Junior Software Engineer", PowerAxis.DOWNWARD.value),
    ]

    for name, role, axis in personas_to_test:
        print("\n" + "-" * 70)
        print(f" [PERSONA] {name} ({role}) | POWER AXIS: {axis}")
        print("-" * 70)

        session = ConversationSession(
            session_id=session_id,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            target_speaker="USER",
            counterpart_name=name,
            counterpart_role=role,
            power_axis=axis,
            dialogue=aligned_dialogue
        )

        evaluation = engine.evaluate_session(session, top_n=2, use_llm=False)

        print(f"  Presence Score:         {evaluation.metrics.presence_score}/100")
        print(f"  Assertiveness Score:    {evaluation.metrics.assertiveness_score}/100")
        print(f"  Active Listening Score: {evaluation.metrics.active_listening_score}/100")
        print(f"  Fillers Detected:       {[f'{f.token}: {f.count}' for f in evaluation.metrics.filler_words_detected]}")
        print(f"\n  Strategy Focus:\n    {evaluation.persona_context}")
        print(f"\n  Strategic Summary:\n    {evaluation.longitudinal_summary}")

        print("\n  Top Strengths (N=2):")
        for idx, s in enumerate(evaluation.top_strengths, 1):
            print(f"    {idx}. {s.observation}")
            print(f"       Quote: \"{s.verbatim_quote}\"")

        print("\n  Areas for Improvement & Coached Rephrasing (N=2):")
        for idx, a in enumerate(evaluation.areas_for_improvement, 1):
            print(f"    {idx}. Critique: {a.critique}")
            print(f"       Original Quote: \"{a.verbatim_quote}\"")
            print(f"       Coached Alternative: \"{a.coached_phrasing}\"")

    # -------------------------------------------------------------------------
    # Stage 6: DPDP Statutory Right to Erasure Protocol
    # -------------------------------------------------------------------------
    print("\n[STAGE 6] Statutory Right to Erasure Verification...")
    erase_res = compliance_mgr.execute_statutory_erasure(session_id)
    print(f"  [ERASED] Status={erase_res['status']}, Session={erase_res['session_id']}, Standard={erase_res['compliance_standard']}")

    print("\n" + "=" * 80)
    print(" [SUCCESS] CORE MVP PIPELINE EXECUTION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline_demo()
