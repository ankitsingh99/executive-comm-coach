"""
Executive Communication Coach - First Live Interactive Demo.
Walks through a realistic corporate Hinglish conversation step-by-step on local device.
"""

import sys
import os
import time

# Ensure path resolution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.schema import ConversationSession, Utterance
from engine.persona_ontology import PowerAxis
from engine.coaching_engine import ExecutiveCoachingEngine
from asr_diarization.vad_gater import AmbientVadGate
from asr_diarization.local_stt_engine import LocalSTTEngine
from privacy.pii_redactor import PIIRedactor
from privacy.dpdp_compliance import DPDPComplianceManager
from config import DATA_DIR


def print_step_header(step_num: int, title: str):
    print("\n" + "=" * 80)
    print(f" STEP {step_num}: {title.upper()}")
    print("=" * 80)
    time.sleep(0.3)


def main():
    print("""
 ╔══════════════════════════════════════════════════════════════════════════════╗
 ║         🚀 ON-DEVICE EXECUTIVE CONVERSATIONAL INTELLIGENCE COACH             ║
 ║                   Live Demonstration & Coaching Pipeline                     ║
 ╚══════════════════════════════════════════════════════════════════════════════╝
""")

    # -------------------------------------------------------------------------
    # STEP 1: Ambient Sensing & Voice Activity Detection (Silero VAD)
    # -------------------------------------------------------------------------
    print_step_header(1, "Low-Power Ambient Acoustic Sensing & Gating")
    print("  Status: Passive Monitoring Mode (< 2.5% battery/hr via 16kHz PCM ring buffer)")
    print("  Acoustic Gate: Evaluates 32ms frames; purges non-speech buffers within 3s.\n")

    vad = AmbientVadGate(speech_prob_threshold=0.75, sustained_window_ms=600.0)

    # Frame 1 & 2: Background ambient noise
    _, msg1 = vad.evaluate_frame(timestamp_ms=0, speech_prob=0.12)
    print(f"  [0ms   - Background Noise]: τ = 0.12 -> {msg1}")
    _, msg2 = vad.evaluate_frame(timestamp_ms=100, speech_prob=0.18)
    print(f"  [100ms - Background Noise]: τ = 0.18 -> {msg2}")

    # Speech onset: Dialogue begins
    print("\n  >> Meeting dialogue detected in room...")
    for t in [200, 300, 400, 500, 600, 700, 800]:
        triggered, msg = vad.evaluate_frame(timestamp_ms=t, speech_prob=0.89)
        if triggered:
            print(f"  [{t}ms - Speech Onset   ]: τ = 0.89 -> 🚨 {msg}")
            break

    # -------------------------------------------------------------------------
    # STEP 2: DPDP Statutory Consent & Audible Notification
    # -------------------------------------------------------------------------
    print_step_header(2, "DPDP Act (2023) Statutory Consent & Safeguards")
    compliance_mgr = DPDPComplianceManager(storage_root=DATA_DIR)
    session_id = f"demo_session_{int(time.time())}"

    chime = compliance_mgr.trigger_audible_chime()
    print(f"  🔔 {chime}")
    consent = compliance_mgr.log_session_consent(session_id, counterpart_notified=True)
    print(f"  📜 Explicit Consent Record Logged: {consent.session_id}")
    print("  🔒 Storage: Local-first AES-256 encrypted vault initialized.")

    # -------------------------------------------------------------------------
    # STEP 3: Local Speech-to-Text & Speaker Diarization
    # -------------------------------------------------------------------------
    print_step_header(3, "Bilingual Hinglish Speech Recognition & Speaker Diarization")

    meeting_dialogue_raw = """
COUNTERPART: Reshma, Vikram here. Can you give me a quick status on the cloud migration and P99 latency?
USER: Yeah so basically, matlab we were looking at the logs and I just think maybe we could finish by Friday, but there were some database blockers.
COUNTERPART: What is the exact quantitative impact on our API latency? Are we going to breach our SLA?
USER: Understood. Our data demonstrates that the P99 latency dropped by 42ms across all regional clusters. We have decided to ship the release branch tomorrow at 10 AM, and project budget is ₹35 lakh.
"""

    stt_engine = LocalSTTEngine()
    utterances = stt_engine.process_local_transcript(meeting_dialogue_raw)

    print("  [Diarized Transcript Turns]:")
    for u in utterances:
        speaker_tag = f"👤 {u.speaker}" if u.speaker == "USER" else f"👔 {u.speaker}"
        print(f"   {speaker_tag} [{u.start_time:04.1f}s - {u.end_time:04.1f}s]: \"{u.transcript}\"")

    # -------------------------------------------------------------------------
    # STEP 4: Local PII & Sensitive Entity Redaction
    # -------------------------------------------------------------------------
    print_step_header(4, "Privacy-by-Design Local PII Redaction")
    print("  Scanning transcript turns for sensitive financial data, credentials, and phone numbers before LLM synthesis...\n")

    redacted_turns = []
    total_redactions = {}
    for u in utterances:
        red_text, counts = PIIRedactor.redact_text(u.transcript)
        redacted_turns.append(Utterance(speaker=u.speaker, start_time=u.start_time, end_time=u.end_time, transcript=red_text))
        for k, v in counts.items():
            total_redactions[k] = total_redactions.get(k, 0) + v

    for u in redacted_turns:
        if "REDACTED" in u.transcript:
            print(f"   🛡️ Scrubbed Turn: \"{u.transcript}\"")
    print(f"   📊 Redaction Summary: {total_redactions}")

    # -------------------------------------------------------------------------
    # STEP 5: Persona Context & Executive Coaching Synthesis
    # -------------------------------------------------------------------------
    print_step_header(5, "Executive Coaching Synthesis (Upward BLUF Evaluation)")
    
    counterpart_name = "Vikram Malhotra"
    counterpart_role = "VP of Engineering"
    power_axis = PowerAxis.UPWARD

    session = ConversationSession(
        session_id=session_id,
        timestamp_utc="2026-08-27T18:55:00Z",
        target_speaker="USER",
        counterpart_name=counterpart_name,
        counterpart_role=counterpart_role,
        power_axis=power_axis.value,
        dialogue=redacted_turns
    )

    coach = ExecutiveCoachingEngine(use_local_only=True)
    evaluation = coach.evaluate_session(session, top_n=2)

    print(f"  🎯 Relational Dynamic: {evaluation.persona_context}\n")
    
    # Quantitative Scores
    print("  ┌────────────────────────────────────────────────────────┐")
    print("  │              📊 EXECUTIVE SCORECARD                    │")
    print("  ├────────────────────────┬───────────────────────────────┤")
    print(f"  │  Executive Presence    │  {evaluation.metrics.presence_score:>3}/100                      │")
    print(f"  │  Assertiveness Index   │  {evaluation.metrics.assertiveness_score:>3}/100                      │")
    print(f"  │  Active Listening      │  {evaluation.metrics.active_listening_score:>3}/100                      │")
    fillers_str = ", ".join([f"{f.token}: {f.count}" for f in evaluation.metrics.filler_words_detected]) or "None"
    print(f"  │  Fillers Detected      │  {fillers_str:<29}│")
    print("  └────────────────────────┴───────────────────────────────┘\n")

    print(f"  📝 Executive Summary:\n     {evaluation.longitudinal_summary}\n")

    print("  🌟 TOP POSITIVE STRENGTHS:")
    for idx, s in enumerate(evaluation.top_strengths, 1):
        print(f"    {idx}. {s.observation}")
        print(f"       💬 Exact Quote: \"{s.verbatim_quote}\"")

    print("\n  📈 AREAS FOR IMPROVEMENT & HIGH-IMPACT COACHED PHRASING:")
    for idx, a in enumerate(evaluation.areas_for_improvement, 1):
        print(f"    {idx}. Critique: {a.critique}")
        print(f"       ❌ What You Said:   \"{a.verbatim_quote}\"")
        print(f"       ✅ Executive BLUF:  \"{a.coached_phrasing}\"")

    # -------------------------------------------------------------------------
    # STEP 6: Statutory Right to Erasure
    # -------------------------------------------------------------------------
    print_step_header(6, "DPDP Act Statutory Right to Erasure Test")
    wipe_res = compliance_mgr.execute_statutory_erasure(session_id)
    print(f"  🗑️ Erasure Result: Status={wipe_res['status']} | Session ID={wipe_res['session_id']}")
    print(f"  🔒 Compliance Rule: {wipe_res['compliance_standard']}")

    print("\n" + "=" * 80)
    print(" 🎉 DEMO COMPLETED SUCCESSFULLY: 100% ON-DEVICE LOCAL EXECUTION")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
