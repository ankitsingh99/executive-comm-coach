"""
Live Hardware Microphone Universal Communication Coach.
Records your real voice from the microphone, transcribes it on-device,
and dynamically coaches your spoken communication across all registers
(Formal/Executive, Collaborative/Peer, Casual/Social, Solo Practice, and Conflict).
"""

import sys
import os
import time
import argparse

# Ensure path resolution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.schema import ConversationSession, Utterance
from engine.persona_ontology import PowerAxis
from engine.coaching_engine import ExecutiveCoachingEngine
from asr_diarization.live_mic_recorder import LiveMicRecorder
from asr_diarization.local_stt_engine import LocalSTTEngine
from asr_diarization.acoustic_speaker_detector import AcousticSpeakerToneDetector
from privacy.pii_redactor import PIIRedactor
from privacy.dpdp_compliance import DPDPComplianceManager
from config import DATA_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Live Microphone Universal Communication Coach")
    parser.add_argument("duration", type=int, nargs="?", default=8, help="Recording duration in seconds (default: 8)")
    parser.add_argument("--axis", type=str, default=None, choices=["SOLO", "CASUAL", "LATERAL", "UPWARD", "DOWNWARD", "CONFLICT"], help="Power Axis / Communication Mode")
    parser.add_argument("--counterpart", type=str, default=None, help="Counterpart Name / Title")
    parser.add_argument("--role", type=str, default=None, help="Counterpart Role")
    parser.add_argument("--non-interactive", action="store_true", help="Skip post-transcription interactive context prompt")
    return parser.parse_args()


def prompt_for_communication_context(detected_count: int = 1, detected_tone: str = "Calm & Measured") -> tuple:
    """Interactively asks the user who they were speaking with after transcription and acoustic analysis."""
    default_choice = "1" if detected_count == 1 else "3"
    rec_label = "(Acoustic recommendation: Solo Practice)" if detected_count == 1 else f"(Acoustic recommendation: Multi-party dialogue, {detected_count} voices detected)"

    print("\n  +--------------------------------------------------------------+")
    print("  |            COMMUNICATION CONTEXT CALIBRATION                 |")
    print("  +--------------------------------------------------------------+")
    print(f"  Acoustic Sensing: {rec_label}")
    print("  Who were you speaking with, or what was the context?\n")
    print(f"    [1] Solo Practice / Monologue (Speaking all by myself, rehearsing speech/thoughts) {'[Default]' if default_choice == '1' else ''}")
    print("    [2] Casual / Social (Friend, informal coffee chat, social banter)")
    print(f"    [3] Collaborative / Peer (Colleague, sync, sprint/project collaboration) {'[Default]' if default_choice == '3' else ''}")
    print("    [4] Formal / Executive (Manager, Director, CXO, interview, proposal)")
    print("    [5] Mentorship / Downward (Direct report, mentee, 1-on-1 coaching)")
    print("    [6] Difficult / Conflict Resolution (Negotiation, tension, debate)")
    print()

    axis_map = {
        "1": (PowerAxis.SOLO, "Self (Solo Practice)", "Self"),
        "2": (PowerAxis.CASUAL, "Friend / Colleague", "Informal Contact"),
        "3": (PowerAxis.LATERAL, "Peer Collaborator", "Team Member"),
        "4": (PowerAxis.UPWARD, "Senior Leadership", "Manager / Executive"),
        "5": (PowerAxis.DOWNWARD, "Direct Report", "Mentee / Team Member"),
        "6": (PowerAxis.CONFLICT, "Counterpart", "Negotiation Contact")
    }

    choice = ""
    if sys.stdin.isatty():
        try:
            choice = input(f"  Select context [1-6, default: {default_choice}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            choice = default_choice
    else:
        choice = default_choice

    if choice not in axis_map:
        choice = default_choice

    selected_axis, default_name, default_role = axis_map[choice]

    if selected_axis == PowerAxis.SOLO:
        return selected_axis, default_name, default_role

    # For non-solo, optionally ask counterpart name if interactive
    counterpart_name = default_name
    counterpart_role = default_role
    if sys.stdin.isatty():
        try:
            custom_name = input(f"  Enter counterpart name/title (optional, press Enter for '{default_name}'): ").strip()
            if custom_name:
                counterpart_name = custom_name
        except (EOFError, KeyboardInterrupt):
            pass

    return selected_axis, counterpart_name, counterpart_role


def main():
    args = parse_args()

    print("""
 +------------------------------------------------------------------------------+
 |           LIVE ON-DEVICE UNIVERSAL COMMUNICATION COACH                       |
 |       Hardware Microphone Sensing & Register-Adaptive Coaching               |
 +------------------------------------------------------------------------------+
""")

    duration = args.duration
    compliance_mgr = DPDPComplianceManager(storage_root=DATA_DIR)
    session_id = f"live_mic_{int(time.time())}"

    # Step 1: DPDP Chime & Consent
    print(" [DPDP NOTICE] Playing statutory recording chime...")
    chime = compliance_mgr.trigger_audible_chime()
    print(f" {chime}")
    compliance_mgr.log_session_consent(session_id, counterpart_notified=True)

    # Step 2: Live Microphone Recording
    print(f"\n [MICROPHONE INGESTION] Recording {duration}s from your microphone...")
    print(" >> Speak now naturally into your microphone...\n")
    
    recorder = LiveMicRecorder()
    wav_path = recorder.record_to_wav(duration_seconds=duration)

    # Step 3: Acoustic Voice & Tone Detection + Local STT
    print("\n [ACOUSTIC SENSING] Analyzing vocal pitch, energy dynamics, and speaker count...")
    acoustic_detector = AcousticSpeakerToneDetector()
    acoustic_result = acoustic_detector.analyze_wav_file(wav_path)

    print(f"\n  +--------------------------------------------------------------+")
    print(f"  |              ACOUSTIC VOICE & TONE DETECTION                 |")
    print(f"  +--------------------------------------------------------------+")
    spk_type = "Solo Speaker" if acoustic_result.detected_speaker_count == 1 else f"Multi-Speaker ({acoustic_result.detected_speaker_count} distinct voices)"
    print(f"  • Detected Voices: {acoustic_result.detected_speaker_count} [{spk_type}]")
    print(f"  • Overall Vocal Tone: {acoustic_result.overall_tone}")
    for spk in acoustic_result.speakers:
        print(f"    - {spk.speaker_id}: Tone: {spk.tone_label} | Pitch: {spk.mean_pitch_hz} Hz (Range: {spk.pitch_range_hz} Hz) | Talk Time: {spk.talk_time_percentage}%")
    print(f"  +--------------------------------------------------------------+\n")

    print(" [ON-DEVICE STT] Transcribing captured speech locally with NVIDIA Parakeet STT (nvidia/parakeet-ctc-0.6b)...")
    stt_engine = LocalSTTEngine()
    utterances = stt_engine.transcribe_audio_file(wav_path, speaker_id="USER")

    if not utterances or not any(u.transcript.strip() for u in utterances):
        print("\n [NOTICE] No speech was detected during the recording window.")
        print(" Please verify your microphone volume and speak closer to the mic.")
        if os.path.exists(wav_path):
            os.remove(wav_path)
        return

    print("\n [YOUR EXACT WORDS AS TRANSCRIBED]:")
    for u in utterances:
        print(f"   >>> \"{u.transcript}\"")

    # Step 4: Privacy Redaction
    redacted_turns = []
    for u in utterances:
        red_text, _ = PIIRedactor.redact_text(u.transcript)
        redacted_turns.append(Utterance(speaker=u.speaker, start_time=u.start_time, end_time=u.end_time, transcript=red_text))

    # Step 5: Post-Transcription Context Inquiry (Don't assume!)
    if not args.non_interactive and args.axis is None:
        axis_enum, counterpart_name, counterpart_role = prompt_for_communication_context(
            detected_count=acoustic_result.detected_speaker_count,
            detected_tone=acoustic_result.overall_tone
        )
    else:
        axis_enum = PowerAxis(args.axis.upper()) if args.axis else (PowerAxis.SOLO if acoustic_result.detected_speaker_count == 1 else PowerAxis.LATERAL)
        counterpart_name = args.counterpart or ("Self (Solo Practice)" if axis_enum == PowerAxis.SOLO else "Counterpart")
        counterpart_role = args.role or ("Self" if axis_enum == PowerAxis.SOLO else "Colleague")

    # Step 6: Dynamic Coaching Synthesis
    print(f"\n [COACHING ANALYSIS] Calibrating feedback for {axis_enum.value} context ({counterpart_name})...")
    session = ConversationSession(
        session_id=session_id,
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        target_speaker="USER",
        counterpart_name=counterpart_name,
        counterpart_role=counterpart_role,
        power_axis=axis_enum.value,
        dialogue=redacted_turns
    )

    coach = ExecutiveCoachingEngine(use_local_only=True)
    evaluation = coach.evaluate_session(session, top_n=None)
    evaluation.metrics.acoustic_analysis = acoustic_result

    card_title = "COMMUNICATION SCORECARD" if axis_enum != PowerAxis.UPWARD else "EXECUTIVE SCORECARD"
    print(f"\n  +--------------------------------------------------------+")
    print(f"  |                  {card_title:<38}|")
    print("  +------------------------+-------------------------------+");
    metric_label = "Presence & Delivery" if axis_enum == PowerAxis.SOLO else "Executive Presence "
    print(f"  |  {metric_label}  |  {evaluation.metrics.presence_score:>3}/100                      |")
    print(f"  |  Assertiveness Index   |  {evaluation.metrics.assertiveness_score:>3}/100                      |")
    reciprocity_label = "Reciprocity / Listen" if axis_enum in [PowerAxis.CASUAL, PowerAxis.LATERAL] else "Active Listening    "
    print(f"  |  {reciprocity_label} |  {evaluation.metrics.active_listening_score:>3}/100                      |")
    fillers_str = ", ".join([f"{f.token}: {f.count}" for f in evaluation.metrics.filler_words_detected]) or "None"
    print(f"  |  Fillers Detected      |  {fillers_str:<29}|")
    print(f"  |  Vocal Tone Profile    |  {acoustic_result.overall_tone:<29}|")
    print("  +------------------------+-------------------------------+\n")

    print(f"  [COACHING TAKEAWAY]:\n     {evaluation.longitudinal_summary}\n")

    print("  TOP POSITIVE STRENGTHS:")
    if not evaluation.top_strengths:
        print("    (No specific delivery highlights for this turn)")
    for idx, s in enumerate(evaluation.top_strengths, 1):
        print(f"    {idx}. {s.observation}")
        print(f"       Quote: \"{s.verbatim_quote}\"")

    rephrase_title = "COACHED REPHRASING & REFINEMENTS:" if axis_enum == PowerAxis.SOLO else "DYNAMIC COACHED REPHRASING:"
    print(f"\n  AREAS FOR IMPROVEMENT & {rephrase_title}")
    if not evaluation.areas_for_improvement:
        print("    • None! Delivery was exceptionally clean with zero detected friction points.")
    for idx, a in enumerate(evaluation.areas_for_improvement, 1):
        print(f"    {idx}. Critique: {a.critique}")
        print(f"       Original Spoken:  \"{a.verbatim_quote}\"")
        rephrase_label = "Polished Phrasing:" if axis_enum == PowerAxis.SOLO else "Coached Delivery: "
        print(f"       {rephrase_label}  \"{a.coached_phrasing}\"")

    # Step 7: Cleanup
    if os.path.exists(wav_path):
        os.remove(wav_path)
    print("\n [COMPLETE] Audio buffer flushed and temporary session secured.")


if __name__ == "__main__":
    main()

