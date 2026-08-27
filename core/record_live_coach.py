"""
Live Hardware Microphone Executive Communication Coach.
Records your real voice from the microphone, transcribes it on-device,
and dynamically coaches your actual spoken words into Executive BLUF phrasing.
"""

import sys
import os
import time

# Ensure path resolution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.schema import ConversationSession, Utterance
from engine.persona_ontology import PowerAxis
from engine.coaching_engine import ExecutiveCoachingEngine
from asr_diarization.live_mic_recorder import LiveMicRecorder
from asr_diarization.local_stt_engine import LocalSTTEngine
from privacy.pii_redactor import PIIRedactor
from privacy.dpdp_compliance import DPDPComplianceManager
from config import DATA_DIR


def main():
    print("""
 +------------------------------------------------------------------------------+
 |       LIVE MICROPHONE ON-DEVICE EXECUTIVE COMMUNICATION COACH                |
 |                  Real Hardware Acoustic Sensing & Analysis                   |
 +------------------------------------------------------------------------------+
""")

    duration = 8
    if len(sys.argv) > 1:
        try:
            duration = int(sys.argv[1])
        except ValueError:
            duration = 8

    counterpart_name = "Vikram Malhotra"
    counterpart_role = "VP of Engineering"
    axis = PowerAxis.UPWARD

    compliance_mgr = DPDPComplianceManager(storage_root=DATA_DIR)
    session_id = f"live_mic_{int(time.time())}"

    # Step 1: DPDP Chime & Consent
    print(" [DPDP NOTICE] Playing statutory recording chime...")
    chime = compliance_mgr.trigger_audible_chime()
    print(f" {chime}")
    compliance_mgr.log_session_consent(session_id, counterpart_notified=True)

    # Step 2: Live Microphone Recording
    print(f"\n [MICROPHONE INGESTION] Recording {duration} seconds from your microphone...")
    print(" >> Speak naturally (e.g. state your project status, blockers, or timeline)...\n")
    
    recorder = LiveMicRecorder()
    wav_path = recorder.record_to_wav(duration_seconds=duration)

    # Step 3: Local Speech-to-Text Transcription (Faster-Whisper on CPU/NPU)
    print("\n [ON-DEVICE STT] Transcribing captured speech locally with Whisper...")
    stt_engine = LocalSTTEngine(model_size="tiny")
    utterances = stt_engine.transcribe_audio_file(wav_path, speaker_id="USER")

    if not utterances or not utterances[0].transcript.strip():
        print(" [WARNING] No distinct speech detected in audio recording.")
        return

    print("\n [YOUR EXACT WORDS AS TRANSCRIBED]:")
    for u in utterances:
        print(f"   >>> \"{u.transcript}\"")

    # Step 4: Privacy Redaction
    redacted_turns = []
    for u in utterances:
        red_text, counts = PIIRedactor.redact_text(u.transcript)
        redacted_turns.append(Utterance(speaker=u.speaker, start_time=u.start_time, end_time=u.end_time, transcript=red_text))

    # Step 5: Dynamic Executive Coaching Synthesis
    print(f"\n [EXECUTIVE ANALYSIS] Calibrating coaching against {axis.value} (BLUF) dynamic...")
    session = ConversationSession(
        session_id=session_id,
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        target_speaker="USER",
        counterpart_name=counterpart_name,
        counterpart_role=counterpart_role,
        power_axis=axis.value,
        dialogue=redacted_turns
    )

    coach = ExecutiveCoachingEngine(use_local_only=True)
    evaluation = coach.evaluate_session(session, top_n=2)

    print("\n  +--------------------------------------------------------+")
    print("  |                  EXECUTIVE SCORECARD                   |")
    print("  +------------------------+-------------------------------+");
    print(f"  |  Executive Presence    |  {evaluation.metrics.presence_score:>3}/100                      |")
    print(f"  |  Assertiveness Index   |  {evaluation.metrics.assertiveness_score:>3}/100                      |")
    print(f"  |  Active Listening      |  {evaluation.metrics.active_listening_score:>3}/100                      |")
    fillers_str = ", ".join([f"{f.token}: {f.count}" for f in evaluation.metrics.filler_words_detected]) or "None"
    print(f"  |  Fillers Detected      |  {fillers_str:<29}|")
    print("  +------------------------+-------------------------------+\n")

    print(f"  [COACHING TAKEAWAY]:\n     {evaluation.longitudinal_summary}\n")

    print("  TOP POSITIVE STRENGTHS:")
    for idx, s in enumerate(evaluation.top_strengths, 1):
        print(f"    {idx}. {s.observation}")
        print(f"       Quote: \"{s.verbatim_quote}\"")

    print("\n  AREAS FOR IMPROVEMENT & DYNAMIC COACHED BLUF REPHRASING:")
    for idx, a in enumerate(evaluation.areas_for_improvement, 1):
        print(f"    {idx}. Critique: {a.critique}")
        print(f"       Original Spoken:  \"{a.verbatim_quote}\"")
        print(f"       Coached BLUF:     \"{a.coached_phrasing}\"")

    # Step 6: Cleanup
    if os.path.exists(wav_path):
        os.remove(wav_path)
    print("\n [COMPLETE] Audio buffer flushed and temporary session secured.")


if __name__ == "__main__":
    main()
