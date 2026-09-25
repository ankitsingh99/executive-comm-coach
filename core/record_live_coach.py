"""
Live Hardware Microphone Universal Communication Coach.
Records your real voice from the microphone, transcribes it on-device,
and dynamically coaches your spoken communication across all registers
(Formal/Executive, Collaborative/Peer, Casual/Social, Solo Practice, and Conflict).
"""

import argparse
import os
import sys
import time
from typing import Optional

# Ensure path resolution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from asr_diarization.acoustic_filler_detector import AcousticFillerDetector
from asr_diarization.acoustic_speaker_detector import AcousticSpeakerToneDetector
from asr_diarization.diarizer import DiarizationEngine
from asr_diarization.gemini_audio_engine import GeminiAudioEngine
from asr_diarization.live_mic_recorder import LiveMicRecorder
from asr_diarization.local_stt_engine import LocalSTTEngine
from asr_diarization.speaker_voiceprint_registry import SpeakerVoiceprintRegistry
from config import DATA_DIR, GEMINI_MODEL, get_gemini_api_key
from engine.coaching_engine import ExecutiveCoachingEngine
from engine.context_calibrator import CommunicationContextCalibrator
from engine.persona_ontology import PowerAxis
from engine.schema import ConversationSession, Utterance
from privacy.dpdp_compliance import DPDPComplianceManager
from privacy.pii_redactor import PIIRedactor


def parse_args():
    parser = argparse.ArgumentParser(description="Live Microphone Universal Communication Coach")
    parser.add_argument(
        "duration",
        type=int,
        nargs="?",
        default=None,
        help="Optional maximum recording duration in seconds (default: dynamic until silence)",
    )
    default_silence = float(os.environ.get("SILENCE_SEC", 2.0))
    parser.add_argument(
        "--silence-sec",
        "--silence",
        "--pause",
        "-s",
        dest="silence_sec",
        type=float,
        default=default_silence,
        help="Silence pause duration in seconds after speech to conclude conversation (default: 2.0s)",
    )
    parser.add_argument(
        "--fixed-duration", action="store_true", help="Force fixed duration recording without waiting for silence"
    )
    parser.add_argument(
        "--sensitivity",
        type=str,
        default="high",
        choices=["high", "medium", "low"],
        help="Microphone ambient pickup sensitivity (default: high)",
    )
    parser.add_argument(
        "--axis",
        type=str,
        default=None,
        choices=["SOLO", "CASUAL", "LATERAL", "UPWARD", "DOWNWARD", "CONFLICT"],
        help="Power Axis / Communication Mode",
    )
    parser.add_argument("--counterpart", type=str, default=None, help="Counterpart Name / Title")
    parser.add_argument("--role", type=str, default=None, help="Counterpart Role")
    parser.add_argument("--gemini-key", type=str, default=None, help="Gemini API Key (optional)")
    parser.add_argument("--local-only", action="store_true", help="Force local on-device models only")
    parser.add_argument(
        "--non-interactive", action="store_true", help="Skip post-transcription interactive context prompt"
    )
    parser.add_argument(
        "--direct", "--now", action="store_true", help="Record immediately without waiting for speech detection"
    )
    parser.add_argument("--ambient", "--listen", action="store_true", help="Explicit ambient monitoring mode")
    parser.add_argument("--list-voices", action="store_true", help="List enrolled voiceprints in local registry")
    parser.add_argument("--delete-voice", type=str, default=None, help="Delete an enrolled voiceprint by name")
    return parser.parse_args()


def prompt_for_communication_context(
    detected_count: int = 1,
    detected_tone: str = "Calm & Measured",
    utterances: Optional[list] = None,
    acoustic_result: Optional[object] = None,
) -> tuple:
    """Interactively asks the user who they were speaking with after transcription and acoustic analysis."""
    # Run multimodal context calibration using acoustic modulation, turn dynamics, and discourse
    inference = CommunicationContextCalibrator.infer_context(acoustic_result, utterances or [])

    axis_to_num = {
        PowerAxis.SOLO: "1",
        PowerAxis.CASUAL: "2",
        PowerAxis.LATERAL: "3",
        PowerAxis.UPWARD: "4",
        PowerAxis.DOWNWARD: "5",
        PowerAxis.CONFLICT: "6",
    }
    default_choice = axis_to_num.get(inference.recommended_axis, "1" if detected_count == 1 else "3")

    print("\n  +--------------------------------------------------------------+")
    print("  |            COMMUNICATION CONTEXT CALIBRATION                 |")
    print("  +--------------------------------------------------------------+")
    print(f"  • Acoustic Dynamics: {inference.acoustic_rationale}")
    print(f"  • Discourse Content: {inference.semantic_rationale}")
    print(
        f"  • Inferred Context:  [{default_choice}] {inference.recommended_axis.value} ({int(inference.confidence_score * 100)}% confidence)"
    )
    print("  Who were you speaking with, or what was the context?\n")
    print(
        f"    [1] Solo Practice / Monologue (Speaking all by myself, rehearsing speech/thoughts) {'[Recommended Default]' if default_choice == '1' else ''}"
    )
    print(
        f"    [2] Casual / Social (Friend, informal coffee chat, social banter) {'[Recommended Default]' if default_choice == '2' else ''}"
    )
    print(
        f"    [3] Collaborative / Peer (Colleague, sync, sprint/project collaboration) {'[Recommended Default]' if default_choice == '3' else ''}"
    )
    print(
        f"    [4] Formal / Executive (Manager, Director, CXO, interview, proposal) {'[Recommended Default]' if default_choice == '4' else ''}"
    )
    print(
        f"    [5] Mentorship / Downward (Direct report, mentee, 1-on-1 coaching) {'[Recommended Default]' if default_choice == '5' else ''}"
    )
    print(
        f"    [6] Difficult / Conflict Resolution (Negotiation, tension, debate) {'[Recommended Default]' if default_choice == '6' else ''}"
    )
    print()

    axis_map = {
        "1": (PowerAxis.SOLO, "Self (Solo Practice)", "Self"),
        "2": (PowerAxis.CASUAL, "Friend / Colleague", "Informal Contact"),
        "3": (PowerAxis.LATERAL, "Peer Collaborator", "Team Member"),
        "4": (PowerAxis.UPWARD, "Senior Leadership", "Manager / Executive"),
        "5": (PowerAxis.DOWNWARD, "Direct Report", "Mentee / Team Member"),
        "6": (PowerAxis.CONFLICT, "Counterpart", "Negotiation Contact"),
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
                custom_name = input(
                    f"  Enter your name (to remember your voice for future auto-recognition, or Enter for '{default_name}'): "
                ).strip()
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
            custom_name = input(
                f"  Enter counterpart name/title (to remember their voice for future auto-recognition, or Enter for '{default_name}'): "
            ).strip()
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
            print(
                f"  • {s['name']:<20} | Role: {s['role']:<15} | Mode: {s['power_axis']:<10} | Pitch: {s['mean_pitch_hz']}Hz"
            )
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
    print(
        f"""
 +------------------------------------------------------------------------------+
 |           LIVE UNIVERSAL COMMUNICATION COACH                                 |
 |       {engine_tag:<71}|
 +------------------------------------------------------------------------------+
"""
    )

    silence_threshold = getattr(args, "silence_sec", 2.0)
    max_duration = args.duration if (args.duration and args.duration > 0) else 300
    compliance_mgr = DPDPComplianceManager(storage_root=DATA_DIR)
    recorder = LiveMicRecorder()

    sensitivity_level = getattr(args, "sensitivity", "high").lower()
    sens_map = {"high": (0.45, 2.0), "medium": (0.55, 1.5), "low": (0.65, 1.0)}
    speech_prob_thresh, gain_val = sens_map.get(sensitivity_level, (0.45, 2.0))

    # Step 0: Ambient Conversation Auto-Detection & Nudge (Default)
    if not getattr(args, "direct", False):
        print(f" [AMBIENT CONVERSATION DETECTOR ACTIVE - Sensitivity: {sensitivity_level.upper()}]")
        print("    Passively monitoring microphone for conversation onset (< 2.5% CPU)...")
        print("    Speak naturally when your conversation starts.\n")

        def on_nudge_callback(prob):
            print(f"\n  [NUDGE] We detected you started speaking! (Confidence: {int(prob * 100)}%)")
            print(
                f"     Recording will continue until conversation finishes (>{silence_threshold}s pause after speech)..."
            )
            return True

        # Passively listen until conversation start is detected
        recorder.listen_for_speech_and_nudge(
            speech_prob_threshold=speech_prob_thresh, gain_boost=gain_val, on_speech_detected_callback=on_nudge_callback
        )

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
            max_duration_sec=max_duration,
            speech_prob_threshold=speech_prob_thresh,
            gain_boost=gain_val,
        )

    # Step 2.5: Acoustic Non-Phonetic Filler & Vocal Hesitation Scan
    print("\n [ACOUSTIC FILLER SCAN] Parsing audio for non-phonetic vocal hesitations (umm, aah, aaaaa, uhh)...")
    filler_detector = AcousticFillerDetector()
    acoustic_fillers = filler_detector.detect_fillers_from_wav(wav_path)
    if acoustic_fillers:
        filler_summary = ", ".join(
            [
                f"{af.token} ({af.duration_sec:.1f}s @ {int(af.start_time // 60):02d}:{int(af.start_time % 60):02d})"
                for af in acoustic_fillers
            ]
        )
        print(f"  • Non-Phonetic Hesitations Detected: {len(acoustic_fillers)} [{filler_summary}]")
    else:
        print("  • Non-Phonetic Hesitations: 0 (Continuous fluent vocalization)")

    # Step 3: Transcription & Acoustic Voice Analysis
    utterances = []
    acoustic_result = None

    if use_gemini:
        print("\n [GEMINI MULTIMODAL SENSING] Transcribing speech & analyzing vocal tone via Gemini...")
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

    # Merge acoustic non-phonetic fillers into transcribed utterances
    if utterances and acoustic_fillers:
        utterances = AcousticFillerDetector.inject_fillers_into_utterances(utterances, acoustic_fillers)

    if not utterances or not any(u.transcript.strip() for u in utterances):
        print("\n [NOTICE] No speech was detected during the recording window.")
        print(" Please verify your microphone volume and speak closer to the mic.")
        if os.path.exists(wav_path):
            os.remove(wav_path)
        return

    if acoustic_result is None:
        acoustic_result = AcousticSpeakerToneDetector().analyze_wav_file(wav_path)

    if acoustic_fillers and acoustic_result:
        acoustic_result.acoustic_fillers = acoustic_fillers

    print("\n  +--------------------------------------------------------------+")
    print("  |              ACOUSTIC VOICE & TONE DETECTION                 |")
    print("  +--------------------------------------------------------------+")
    spk_type = (
        "Solo Speaker"
        if acoustic_result.detected_speaker_count == 1
        else f"Multi-Speaker ({acoustic_result.detected_speaker_count} distinct voices)"
    )
    print(f"  • Detected Voices: {acoustic_result.detected_speaker_count} [{spk_type}]")
    print(f"  • Overall Vocal Tone: {acoustic_result.overall_tone}")
    for spk in acoustic_result.speakers:
        print(
            f"    - {spk.speaker_id}: Tone: {spk.tone_label} | Pitch: {spk.mean_pitch_hz} Hz | Talk Time: {spk.talk_time_percentage}%"
        )
    print("  +--------------------------------------------------------------+\n")

    # Step 3.5: Acoustic Voiceprint Identification & Verbal Self-Introduction Detection
    recognized_voice = None
    voice_match_conf = 0.0

    # A. Acoustic Voiceprint Check (works for both Solo and Multi-speaker audio)
    id_res = voice_registry.identify_speaker(wav_path)
    if id_res is not None:
        recognized_voice, voice_match_conf = id_res
        if (
            recognized_voice.is_user
            or recognized_voice.power_axis == "SOLO"
            or acoustic_result.detected_speaker_count == 1
        ):
            print(
                f"  [USER VOICE RECOGNIZED] Welcome back, '{recognized_voice.speaker_name}'! [Identified App User with {int(voice_match_conf * 100)}% match confidence]\n"
            )
        else:
            print(
                f"  [COUNTERPART VOICE RECOGNIZED] Identified Interlocutor: '{recognized_voice.speaker_name}' ({recognized_voice.role}) [Match Confidence: {int(voice_match_conf * 100)}%]"
            )
            print(
                f"     Auto-calibrated relational context to {recognized_voice.power_axis} mode without manual tagging!\n"
            )

    # B. Verbal Self-Introduction Check (e.g. "Hey I am Rahul and today...", "Vikram here", etc.)
    utterances, intro_counterpart, intro_user = DiarizationEngine.detect_and_apply_verbal_introductions(
        utterances, user_speaker_id="USER"
    )

    current_user_name = (
        recognized_voice.speaker_name
        if (recognized_voice and (recognized_voice.is_user or recognized_voice.power_axis == "SOLO"))
        else (intro_user or None)
    )
    current_counterpart_name = (
        recognized_voice.speaker_name
        if (recognized_voice and not recognized_voice.is_user and recognized_voice.power_axis != "SOLO")
        else (intro_counterpart or None)
    )

    if intro_user and not (recognized_voice and recognized_voice.speaker_name == intro_user):
        print(f"  [VERBAL INTRODUCTION DETECTED] Welcome '{intro_user}'! Identified user name from speech.")
        print(f"     Enrolled user voice profile for '{intro_user}' into local memory for future sessions!\n")
        voice_registry.enroll_speaker(
            name=intro_user, role="Self", power_axis="SOLO", audio_signal_or_wav_path=wav_path, is_user=True
        )

    if intro_counterpart and not (recognized_voice and recognized_voice.speaker_name == intro_counterpart):
        print(f"  [VERBAL INTRODUCTION DETECTED] Interlocutor introduced themselves: '{intro_counterpart}'")
        print(f"     Auto-tagged speaker turns and enrolled voiceprint for '{intro_counterpart}' into voice memory!\n")
        voice_registry.enroll_speaker(
            name=intro_counterpart,
            role="Collaborator",
            power_axis="LATERAL",
            audio_signal_or_wav_path=wav_path,
            is_user=False,
        )

    utterances = DiarizationEngine.assign_roles(
        utterances,
        user_speaker_id="USER",
        recognized_counterpart_name=current_counterpart_name,
        recognized_user_name=current_user_name,
    )

    print("  +--------------------------------------------------------------+")
    print("  |              TRANSCRIBED DIALOGUE & SPEAKER TURNS            |")
    print("  +--------------------------------------------------------------+")
    print(
        DiarizationEngine.format_dialogue_cli(
            utterances, user_name=current_user_name, counterpart_name=current_counterpart_name
        )
    )
    print("  +--------------------------------------------------------------+\n")

    # Step 4: Privacy Redaction
    redacted_turns = []
    for u in utterances:
        red_text, _ = PIIRedactor.redact_text(u.transcript)
        redacted_turns.append(
            Utterance(speaker=u.speaker, start_time=u.start_time, end_time=u.end_time, transcript=red_text)
        )

    # Step 5: Post-Transcription Context Resolution (Auto if Recognized/Introduced, else Prompt)
    if recognized_voice is not None and not recognized_voice.is_user and args.axis is None:
        try:
            axis_enum = PowerAxis(recognized_voice.power_axis.upper())
        except Exception:
            axis_enum = PowerAxis.LATERAL
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
            detected_tone=acoustic_result.overall_tone,
            utterances=utterances,
            acoustic_result=acoustic_result,
        )
    else:
        axis_enum = (
            PowerAxis(args.axis.upper())
            if args.axis
            else (PowerAxis.SOLO if acoustic_result.detected_speaker_count == 1 else PowerAxis.LATERAL)
        )
        counterpart_name = args.counterpart or (
            "Self (Solo Practice)" if axis_enum == PowerAxis.SOLO else "Counterpart"
        )
        counterpart_role = args.role or ("Self" if axis_enum == PowerAxis.SOLO else "Colleague")

    # Step 5.5: Proactive User & Counterpart Voice Profiling
    if sys.stdin.isatty() and not args.non_interactive:
        try:
            # 1. Profile and name the ACTIVE USER's voice if not yet enrolled/recognized
            if not current_user_name:
                user_target = input(
                    "\n  [USER VOICE PROFILING] Enter YOUR name to remember your voice profile across sessions (or Enter to skip): "
                ).strip()
                if user_target:
                    voice_registry.enroll_speaker(
                        name=user_target,
                        role="Self",
                        power_axis="SOLO",
                        audio_signal_or_wav_path=wav_path,
                        is_user=True,
                    )
                    current_user_name = user_target
                    if axis_enum == PowerAxis.SOLO:
                        counterpart_name = user_target
                    print(
                        f"  >> [USER VOICE SAVED] Enrolled biometric voice profile for '{user_target}' (App User) into local memory!\n"
                    )

            # 2. Profile and name the COUNTERPART's voice in multi-speaker / relational mode
            if axis_enum != PowerAxis.SOLO and acoustic_result.detected_speaker_count > 1:
                cp_target = (
                    counterpart_name
                    if counterpart_name not in ["Counterpart", "Colleague", "Peer Collaborator", "Friend / Colleague"]
                    else ""
                )
                if not cp_target and not current_counterpart_name:
                    cp_target = input(
                        "  [COUNTERPART VOICE PROFILING] Enter counterpart's name to remember their voice for future auto-tagging (or Enter to skip): "
                    ).strip()
                if cp_target and not current_counterpart_name:
                    voice_registry.enroll_speaker(
                        name=cp_target,
                        role=counterpart_role,
                        power_axis=axis_enum.value,
                        audio_signal_or_wav_path=wav_path,
                        is_user=False,
                    )
                    counterpart_name = cp_target
                    current_counterpart_name = cp_target
                    print(
                        f"  >> [COUNTERPART VOICE SAVED] Enrolled biometric voiceprint for '{cp_target}' ({counterpart_role}) into local memory!\n"
                    )
        except (EOFError, KeyboardInterrupt):
            pass

    # Step 6: Dynamic Coaching Synthesis
    print(f"\n [COACHING ANALYSIS] Calibrating feedback for {axis_enum.value} context ({counterpart_name})...")
    session = ConversationSession(
        session_id=session_id,
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        target_speaker="USER",
        counterpart_name=counterpart_name,
        counterpart_role=counterpart_role,
        power_axis=axis_enum.value,
        dialogue=redacted_turns,
    )

    coach = ExecutiveCoachingEngine(use_local_only=args.local_only)
    evaluation = coach.evaluate_session(session, top_n=None)
    evaluation.metrics.acoustic_analysis = acoustic_result

    card_title = "COMMUNICATION SCORECARD" if axis_enum != PowerAxis.UPWARD else "EXECUTIVE SCORECARD"
    print("\n  +--------------------------------------------------------+")
    print(f"  |                  {card_title:<38}|")

    print("  +------------------------+-------------------------------+")
    metric_label = "Presence & Delivery" if axis_enum == PowerAxis.SOLO else "Executive Presence "
    print(f"  |  {metric_label}  |  {evaluation.metrics.presence_score:>3}/100                      |")
    print(f"  |  Assertiveness Index   |  {evaluation.metrics.assertiveness_score:>3}/100                      |")
    reciprocity_label = (
        "Reciprocity / Listen" if axis_enum in [PowerAxis.CASUAL, PowerAxis.LATERAL] else "Active Listening    "
    )
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
        print(f'       Quote: "{s.verbatim_quote}"')

    rephrase_title = (
        "COACHED REPHRASING & REFINEMENTS:" if axis_enum == PowerAxis.SOLO else "DYNAMIC COACHED REPHRASING:"
    )
    print(f"\n  AREAS FOR IMPROVEMENT & {rephrase_title}")
    if not evaluation.areas_for_improvement:
        print("    • None! Delivery was exceptionally clean with zero detected friction points.")
    for idx, a in enumerate(evaluation.areas_for_improvement, 1):
        print(f"    {idx}. Critique: {a.critique}")
        print(f'       Original Spoken:  "{a.verbatim_quote}"')
        rephrase_label = "Polished Phrasing:" if axis_enum == PowerAxis.SOLO else "Coached Delivery: "
        print(f'       {rephrase_label}  "{a.coached_phrasing}"')

    # Conversational Dynamics & Pacing
    if getattr(evaluation, "dynamics", None):
        dyn = evaluation.dynamics
        print("  +--------------------------------------------------------------+")
        print("  |                 CONVERSATIONAL DYNAMICS & PACING             |")
        print("  +--------------------------------------------------------------+")
        print(
            f"  |  Talk-Time Split       |  USER: {dyn.user_talk_time_pct:.1f}%  |  CP: {dyn.counterpart_talk_time_pct:.1f}%"
        )
        print(
            f"  |  Word Volume Ratio     |  USER: {dyn.user_words_total} words | CP: {dyn.counterpart_words_total} words"
        )
        print(f"  |  Turn Response Latency |  {dyn.average_turn_latency_ms:.0f} ms average")
        print(
            f"  |  Inquiry vs Advocacy   |  {dyn.ask_vs_tell_ratio:.2f}:1 ({dyn.inquiry_count} Asks / {dyn.directive_count} Tells)"
        )
        print(f"  |  Brevity Potential     |  {dyn.brevity_potential_pct:.1f}% potential word reduction")
        print(f"  |  Deep Reflection Score |  {dyn.deep_listening_score}/100 active listening depth")
        print(f"  |  Vocal Tension Index   |  {dyn.vocal_tension_index}")
        print("  +--------------------------------------------------------------+\n")

    # Emotional Trajectory Arc
    if getattr(evaluation, "emotional_trajectory", []):
        print("  +--------------------------------------------------------------+")
        print("  |                 EMOTIONAL TRAJECTORY & TENSION ARC           |")
        print("  +--------------------------------------------------------------+")
        for idx, pt in enumerate(evaluation.emotional_trajectory, 1):
            val_sign = "+" if pt.valence_score >= 0 else ""
            tension_flag = f"[{pt.tension_level} Tension]" if pt.tension_level in ["MEDIUM", "HIGH"] else "[Grounded]"
            print(
                f"    [{idx}] @ {pt.timestamp_sec:4.1f}s ({pt.speaker}): {pt.emotion_label:<24} | Val: {val_sign}{pt.valence_score:.2f} | {pt.pacing_wpm:.0f} wpm {tension_flag}"
            )
        print("  +--------------------------------------------------------------+\n")

    # Consensus & Agreed Outcomes
    if getattr(evaluation, "agreements", []):
        print("  +--------------------------------------------------------------+")
        print("  |                 AGREED OUTCOMES & CONSENSUS POINTS           |")
        print("  +--------------------------------------------------------------+")
        for idx, ag in enumerate(evaluation.agreements, 1):
            print(f"    [{idx}] {ag.headline}")
            print(f"       • Agreed Solution: {ag.agreed_solution}")
            print(f'       • Spoken Quote:   "{ag.verbatim_quote}"')
        print("  +--------------------------------------------------------------+\n")

    # Unresolved Tensions & Open Loops
    if getattr(evaluation, "unresolved_loops", []):
        print("  +--------------------------------------------------------------+")
        print("  |                 UNRESOLVED TENSIONS & OPEN LOOPS             |")
        print("  +--------------------------------------------------------------+")
        for idx, ol in enumerate(evaluation.unresolved_loops, 1):
            print(f"    [{idx}] {ol.concern_topic} (Raised by {ol.raised_by})")
            print(f"       • Context:     {ol.context}")
            print(f"       • Action Plan: {ol.recommended_followup}")
        print("  +--------------------------------------------------------------+\n")

    print("  +--------------------------------------------------------------+")
    print("  |                    KEY HIGHLIGHTS & TAKEAWAYS                |")
    print("  +--------------------------------------------------------------+")
    if not getattr(evaluation, "key_highlights", []):
        print("    * (No explicit strategic milestones or decisions detected in this turn)")
    else:
        for idx, kh in enumerate(evaluation.key_highlights, 1):
            imp_tag = " [High Priority]" if kh.importance == "High" else ""
            print(f"    [{idx}] [{kh.category.upper()}]{imp_tag} {kh.headline}")
            print(f"       • Takeaway: {kh.takeaway}")
            print(f'       • Speaker:  {kh.speaker} (Quote: "{kh.verbatim_quote}")')

    print("\n  +--------------------------------------------------------------+")
    print("  |               DETECTED ACTION ITEMS & COMMITMENTS            |")
    print("  +--------------------------------------------------------------+")
    if not evaluation.action_items:
        print("    * No explicit action items, deadlines, or scheduling commitments detected.")
    else:
        for idx, item in enumerate(evaluation.action_items, 1):
            owner_tag = f"[{item.owner.upper()}]" if item.owner != "USER" else "[USER / YOU]"
            due_str = f" | Due: {item.due_time_or_date}" if item.due_time_or_date else ""
            ampm_str = f" ({item.target_time_inferred_ampm})" if item.target_time_inferred_ampm else ""
            urgency_str = f" [{item.urgency} Urgency]" if item.urgency == "High" else ""
            print(f"    [{idx}] {owner_tag} {item.category}{due_str}{ampm_str}{urgency_str}")
            print(f"       • Task:  {item.task}")
            print(f'       • Quote: "{item.verbatim_quote}"')
    print("  +--------------------------------------------------------------+\n")

    # Step 7: Cleanup
    if os.path.exists(wav_path):
        os.remove(wav_path)
    print(" [COMPLETE] Audio buffer flushed and temporary session secured.")


if __name__ == "__main__":
    main()
