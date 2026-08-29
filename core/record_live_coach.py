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
from asr_diarization.gemini_audio_engine import GeminiAudioEngine
from asr_diarization.diarizer import DiarizationEngine
from asr_diarization.speaker_voiceprint_registry import SpeakerVoiceprintRegistry
from privacy.pii_redactor import PIIRedactor
from privacy.dpdp_compliance import DPDPComplianceManager
from config import DATA_DIR, get_gemini_api_key, GEMINI_MODEL


def parse_args():
    parser = argparse.ArgumentParser(description="Live Microphone Universal Communication Coach")
    parser.add_argument("duration", type=int, nargs="?", default=None, help="Optional maximum recording duration in seconds (default: dynamic until silence)")
    parser.add_argument("--silence-sec", type=float, default=2.2, help="Silence pause duration in seconds after last word to conclude conversation (default: 2.2s)")
    parser.add_argument("--fixed-duration", action="store_true", help="Force fixed duration recording without waiting for silence")
    parser.add_argument("--axis", type=str, default=None, choices=["SOLO", "CASUAL", "LATERAL", "UPWARD", "DOWNWARD", "CONFLICT"], help="Power Axis / Communication Mode")
    parser.add_argument("--counterpart", type=str, default=None, help="Counterpart Name / Title")
    parser.add_argument("--role", type=str, default=None, help="Counterpart Role")
    parser.add_argument("--gemini-key", type=str, default=None, help="Gemini API Key (optional)")
    parser.add_argument("--local-only", action="store_true", help="Force local on-device models only")
    parser.add_argument("--non-interactive", action="store_true", help="Skip post-transcription interactive context prompt")
    parser.add_argument("--direct", "--now", action="store_true", help="Record immediately without waiting for speech detection")
    parser.add_argument("--ambient", "--listen", action="store_true", help="Explicit ambient monitoring mode")
    parser.add_argument("--list-voices", action="store_true", help="List enrolled voiceprints in local registry")
    parser.add_argument("--delete-voice", type=str, default=None, help="Delete an enrolled voiceprint by name")
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
        speaker_name = default_name
        if sys.stdin.isatty():
            try:
                custom_name = input(f"  Enter your name (optional, press Enter for '{default_name}'): ").strip()
                if custom_name:
                    speaker_name = custom_name
            except (EOFError, KeyboardInterrupt):
                pass
        return selected_axis, speaker_name, "Self"

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
    voice_registry = SpeakerVoiceprintRegistry()

    # CLI Utility: List enrolled voiceprints
    if args.list_voices:
        speakers = voice_registry.list_enrolled_speakers()
        print("\n  +--------------------------------------------------------------+")
        print("  |            ENROLLED SPEAKER VOICEPRINT REGISTRY              |")
        print("  +--------------------------------------------------------------+")
        if not speakers:
            print("  (No speaker voiceprints currently enrolled in local vault)")
        for s in speakers:
            print(f"  • {s['name']:<20} | Role: {s['role']:<15} | Mode: {s['power_axis']:<10} | Pitch: {s['mean_pitch_hz']}Hz")
        print("  +--------------------------------------------------------------+\n")
        return

    # CLI Utility: Delete an enrolled voiceprint
    if args.delete_voice:
        deleted = voice_registry.delete_voiceprint(args.delete_voice)
        if deleted:
            print(f"\n  [DPDP ERASURE] Successfully deleted voiceprint for '{args.delete_voice}'.\n")
        else:
            print(f"\n  [NOTICE] No voiceprint found matching '{args.delete_voice}'.\n")
        return

    # Configure Gemini API Key if provided via args or interactive
    if args.gemini_key:
        os.environ["GEMINI_API_KEY"] = args.gemini_key

    gemini_key = get_gemini_api_key()
    if not gemini_key and not args.local_only and sys.stdin.isatty():
        try:
            print("\n  [GEMINI SETUP] Tip: You can use Google Gemini for SOTA speech & vocal tone sensing.")
            entered_key = input("  Enter your GEMINI_API_KEY (or press Enter to run locally): ").strip()
            if entered_key:
                os.environ["GEMINI_API_KEY"] = entered_key
                env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
                with open(env_path, "a", encoding="utf-8") as f:
                    f.write(f"\nGEMINI_API_KEY={entered_key}\n")
                print("  >> Saved GEMINI_API_KEY to .env successfully!\n")
        except (EOFError, KeyboardInterrupt):
            pass

    gemini_engine = GeminiAudioEngine()
    use_gemini = not args.local_only and gemini_engine.is_available()

    engine_tag = f"Powered by Google Gemini ({GEMINI_MODEL})" if use_gemini else "Running On-Device (Local Models)"
    print(f"""
 +------------------------------------------------------------------------------+
 |           LIVE UNIVERSAL COMMUNICATION COACH                                 |
 |       {engine_tag:<71}|
 +------------------------------------------------------------------------------+
""")

    silence_threshold = getattr(args, "silence_sec", 2.2)
    max_duration = args.duration if (args.duration and args.duration > 0) else 180
    compliance_mgr = DPDPComplianceManager(storage_root=DATA_DIR)
    recorder = LiveMicRecorder()

    # Step 0: Ambient Conversation Auto-Detection & Nudge (Default)
    if not getattr(args, "direct", False):
        print(" 👂 [AMBIENT CONVERSATION DETECTOR ACTIVE]")
        print("    Passively monitoring microphone for conversation onset (< 2.5% CPU)...")
        print("    👉 Speak naturally when your conversation starts.\n")
        
        def on_nudge_callback(prob):
            print(f"\n  ✨ [NUDGE] We detected you started speaking! (Confidence: {int(prob * 100)}%)")
            print(f"     Recording will continue until conversation finishes (>{silence_threshold}s pause after speech)...")
            return True

        # Passively listen until conversation start is detected
        recorder.listen_for_speech_and_nudge(on_speech_detected_callback=on_nudge_callback)

    # Step 1: DPDP Chime & Consent
    session_id = f"live_mic_{int(time.time())}"
    print("\n [DPDP NOTICE] Playing statutory recording chime...")
    chime = compliance_mgr.trigger_audible_chime()
    print(f" {chime}")
    compliance_mgr.log_session_consent(session_id, counterpart_notified=True)

    # Step 2: Live Dynamic Microphone Recording (until silence after last word)
    if getattr(args, "fixed_duration", False) and args.duration:
        print(f"\n [MICROPHONE INGESTION] Recording fixed {args.duration}s from your microphone...")
        wav_path = recorder.record_to_wav(duration_seconds=args.duration)
    else:
        wav_path = recorder.record_until_silence(
            silence_threshold_sec=silence_threshold,
            max_duration_sec=max_duration
        )

    # Step 3: Transcription & Acoustic Voice Analysis
    utterances = []
    acoustic_result = None

    if use_gemini:
        print(f"\n [GEMINI MULTIMODAL SENSING] Transcribing speech & analyzing vocal tone via Gemini...")
        utterances, acoustic_result = gemini_engine.process_audio(wav_path, speaker_id="USER")

    # Fallback to local if Gemini was not available or returned empty
    if not utterances or not any(u.transcript.strip() for u in utterances):
        if use_gemini:
            print(" [FALLBACK] Reverting to local acoustic models...")
        print("\n [ACOUSTIC SENSING] Analyzing vocal pitch, energy dynamics, and speaker count...")
        acoustic_detector = AcousticSpeakerToneDetector()
        acoustic_result = acoustic_detector.analyze_wav_file(wav_path)

        print(" [ON-DEVICE STT] Transcribing captured speech locally...")
        stt_engine = LocalSTTEngine()
        utterances = stt_engine.transcribe_audio_file(wav_path, speaker_id="USER")

    if not utterances or not any(u.transcript.strip() for u in utterances):
        print("\n [NOTICE] No speech was detected during the recording window.")
        print(" Please verify your microphone volume and speak closer to the mic.")
        if os.path.exists(wav_path):
            os.remove(wav_path)
        return

    if acoustic_result is None:
        acoustic_result = AcousticSpeakerToneDetector().analyze_wav_file(wav_path)

    print(f"\n  +--------------------------------------------------------------+")
    print(f"  |              ACOUSTIC VOICE & TONE DETECTION                 |")
    print(f"  +--------------------------------------------------------------+")
    spk_type = "Solo Speaker" if acoustic_result.detected_speaker_count == 1 else f"Multi-Speaker ({acoustic_result.detected_speaker_count} distinct voices)"
    print(f"  • Detected Voices: {acoustic_result.detected_speaker_count} [{spk_type}]")
    print(f"  • Overall Vocal Tone: {acoustic_result.overall_tone}")
    for spk in acoustic_result.speakers:
        print(f"    - {spk.speaker_id}: Tone: {spk.tone_label} | Pitch: {spk.mean_pitch_hz} Hz | Talk Time: {spk.talk_time_percentage}%")
    print(f"  +--------------------------------------------------------------+\n")

    # Step 3.5: Acoustic Voiceprint Identification & Verbal Self-Introduction Detection
    recognized_voice = None
    voice_match_conf = 0.0

    # A. Acoustic Voiceprint Check (works for both Solo and Multi-speaker audio)
    id_res = voice_registry.identify_speaker(wav_path)
    if id_res is not None:
        recognized_voice, voice_match_conf = id_res
        if recognized_voice.power_axis == "SOLO" or acoustic_result.detected_speaker_count == 1:
            print(f"  ✨ [VOICEPRINT RECOGNIZED] Welcome back, '{recognized_voice.speaker_name}'! [Match Confidence: {int(voice_match_conf * 100)}%]")
            print(f"     Auto-calibrated profile for {recognized_voice.speaker_name} (Solo Practice).\n")
        else:
            print(f"  ✨ [VOICEPRINT RECOGNIZED] Identified Counterpart: '{recognized_voice.speaker_name}' ({recognized_voice.role}) [Match Confidence: {int(voice_match_conf * 100)}%]")
            print(f"     Auto-calibrated relational context to {recognized_voice.power_axis} mode without manual tagging!\n")

    # B. Verbal Self-Introduction Check (e.g. "Hey I am Rahul and today...", "Vikram here", etc.)
    utterances, intro_counterpart, intro_user = DiarizationEngine.detect_and_apply_verbal_introductions(utterances, user_speaker_id="USER")
    
    current_user_name = recognized_voice.speaker_name if (recognized_voice and recognized_voice.power_axis == "SOLO") else (intro_user or None)
    current_counterpart_name = recognized_voice.speaker_name if (recognized_voice and recognized_voice.power_axis != "SOLO") else (intro_counterpart or None)

    if intro_user and not (recognized_voice and recognized_voice.speaker_name == intro_user):
        print(f"  ✨ [VERBAL INTRODUCTION DETECTED] Welcome '{intro_user}'! Identified speaker name from speech.")
        print(f"     Enrolled voiceprint for '{intro_user}' into local memory for future solo sessions!\n")
        voice_registry.enroll_speaker(
            name=intro_user,
            role="Self",
            power_axis="SOLO",
            audio_signal_or_wav_path=wav_path
        )

    if intro_counterpart and not (recognized_voice and recognized_voice.speaker_name == intro_counterpart):
        print(f"  ✨ [VERBAL INTRODUCTION DETECTED] Interlocutor introduced themselves: '{intro_counterpart}'")
        print(f"     Auto-tagged speaker turns and enrolled voiceprint for '{intro_counterpart}' into voice memory!\n")
        voice_registry.enroll_speaker(
            name=intro_counterpart,
            role="Collaborator",
            power_axis="LATERAL",
            audio_signal_or_wav_path=wav_path
        )

    utterances = DiarizationEngine.assign_roles(
        utterances,
        user_speaker_id="USER",
        recognized_counterpart_name=current_counterpart_name,
        recognized_user_name=current_user_name
    )

    print(f"  +--------------------------------------------------------------+")
    print(f"  |              TRANSCRIBED DIALOGUE & SPEAKER TURNS            |")
    print(f"  +--------------------------------------------------------------+")
    print(DiarizationEngine.format_dialogue_cli(utterances, user_name=current_user_name, counterpart_name=current_counterpart_name))
    print(f"  +--------------------------------------------------------------+\n")

    # Step 4: Privacy Redaction
    redacted_turns = []
    for u in utterances:
        red_text, _ = PIIRedactor.redact_text(u.transcript)
        redacted_turns.append(Utterance(speaker=u.speaker, start_time=u.start_time, end_time=u.end_time, transcript=red_text))

    # Step 5: Post-Transcription Context Resolution (Auto if Recognized/Introduced, else Prompt)
    if recognized_voice is not None and args.axis is None:
        try:
            axis_enum = PowerAxis(recognized_voice.power_axis.upper())
        except Exception:
            axis_enum = PowerAxis.SOLO if recognized_voice.power_axis == "SOLO" else PowerAxis.LATERAL
        counterpart_name = recognized_voice.speaker_name
        counterpart_role = recognized_voice.role
    elif intro_user and args.axis is None and acoustic_result.detected_speaker_count == 1:
        axis_enum = PowerAxis.SOLO
        counterpart_name = intro_user
        counterpart_role = "Self"
    elif intro_counterpart and args.axis is None:
        axis_enum = PowerAxis.LATERAL
        counterpart_name = intro_counterpart
        counterpart_role = "Collaborator"
    elif not args.non_interactive and args.axis is None:
        axis_enum, counterpart_name, counterpart_role = prompt_for_communication_context(
            detected_count=acoustic_result.detected_speaker_count,
            detected_tone=acoustic_result.overall_tone
        )
        # Offer voiceprint enrollment for non-default names
        if sys.stdin.isatty() and counterpart_name not in ["Counterpart", "Self (Solo Practice)", "Self"]:
            try:
                enroll_ans = input(f"\n  [VOICEPRINT MEMORY] Would you like to remember {counterpart_name}'s voice for future auto-recognition? [Y/n]: ").strip().lower()
                if enroll_ans in ["", "y", "yes"]:
                    voice_registry.enroll_speaker(
                        name=counterpart_name,
                        role=counterpart_role,
                        power_axis=axis_enum.value,
                        audio_signal_or_wav_path=wav_path
                    )
                    print(f"  >> Enrolled voiceprint for '{counterpart_name}' into local secure memory!\n")
            except (EOFError, KeyboardInterrupt):
                pass
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

    coach = ExecutiveCoachingEngine(use_local_only=args.local_only)
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

    print(f"\n  +--------------------------------------------------------------+")
    print(f"  |               DETECTED ACTION ITEMS & COMMITMENTS            |")
    print(f"  +--------------------------------------------------------------+")
    if not evaluation.action_items:
        print("    • No explicit action items, deadlines, or scheduling commitments detected.")
    else:
        for idx, item in enumerate(evaluation.action_items, 1):
            owner_tag = f"[{item.owner.upper()}]" if item.owner != "USER" else "[USER / YOU]"
            due_str = f" | Due: {item.due_time_or_date}" if item.due_time_or_date else ""
            urgency_str = f" [{item.urgency} Urgency]" if item.urgency == "High" else ""
            print(f"    📌 {idx}. {owner_tag} {item.category}{due_str}{urgency_str}")
            print(f"       • Task:  {item.task}")
            print(f"       • Quote: \"{item.verbatim_quote}\"")
    print(f"  +--------------------------------------------------------------+\n")

    # Step 7: Cleanup
    if os.path.exists(wav_path):
        os.remove(wav_path)
    print(" [COMPLETE] Audio buffer flushed and temporary session secured.")


if __name__ == "__main__":
    main()

