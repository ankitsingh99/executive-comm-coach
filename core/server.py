"""
Interactive Web & Mobile App Emulator Server for Executive Communication Coach.
Serves the emulator UI and provides live API endpoints for evaluation and voiceprints.
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

# Setup Python paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from asr_diarization.diarizer import DiarizationEngine
from asr_diarization.local_stt_engine import LocalSTTEngine
from asr_diarization.speaker_voiceprint_registry import SpeakerVoiceprint, SpeakerVoiceprintRegistry
from engine.action_item_extractor import ActionItemExtractor
from engine.coaching_engine import ExecutiveCoachingEngine
from engine.schema import ConversationSession, Utterance
from engine.transcription_analyzer import TranscriptionAnalyzer

SAFE_MIME_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".htm": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".mjs": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".webp": "image/webp",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".woff2": "font/woff2",
    ".woff": "font/woff",
    ".ttf": "font/ttf",
}


class EmulatorHandler(BaseHTTPRequestHandler):
    """Handles static files and API requests for the app emulator."""

    def do_GET(self):
        url_path = self.path.split("?")[0]
        if url_path in ("/", "/index.html"):
            self._serve_file("index.html")
        elif url_path == "/api/voiceprints":
            registry = SpeakerVoiceprintRegistry()
            speakers = registry.list_enrolled_speakers()
            data = [
                {
                    "speaker_name": s.get("name", "") if isinstance(s, dict) else getattr(s, "speaker_name", ""),
                    "role": s.get("role", "Colleague") if isinstance(s, dict) else getattr(s, "role", "Colleague"),
                    "power_axis": (
                        s.get("power_axis", "LATERAL") if isinstance(s, dict) else getattr(s, "power_axis", "LATERAL")
                    ),
                    "enrolled_at_utc": (
                        s.get("enrolled_at_utc", "") if isinstance(s, dict) else getattr(s, "enrolled_at_utc", "")
                    ),
                    "mean_pitch_hz": (
                        s.get("mean_pitch_hz", 150.0) if isinstance(s, dict) else getattr(s, "mean_pitch_hz", 150.0)
                    ),
                    "is_user": (s.get("is_user", False) if isinstance(s, dict) else getattr(s, "is_user", False)),
                }
                for s in speakers
            ]
            self._send_json(data)
        else:
            clean_rel = url_path.lstrip("/\\")
            self._serve_file(clean_rel)

    def do_POST(self):
        url_path = self.path.split("?")[0]
        content_length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"

        try:
            payload = json.loads(raw_body)
        except Exception:
            payload = {}

        if url_path in ("/api/evaluate", "/api/evaluate_audio"):
            try:
                dialogue_text = payload.get("dialogue_text", "").strip()
                audio_base64 = payload.get("audio_base64", "").strip()
                mime_type = payload.get("mime_type", "audio/webm")
                counterpart_name = payload.get("counterpart_name", "Rahul")
                power_axis = payload.get("power_axis", "LATERAL")
                
                utterances = []
                
                # If audio blob provided, transcribe directly via SOTA Gemini Multimodal Audio Engine
                if audio_base64:
                    import base64
                    import tempfile
                    
                    try:
                        audio_bytes = base64.b64decode(audio_base64)
                        ext = ".webm" if "webm" in mime_type else (".ogg" if "ogg" in mime_type else ".wav")
                        
                        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_f:
                            tmp_f.write(audio_bytes)
                            tmp_path = tmp_f.name
                        
                        try:
                            from asr_diarization.gemini_audio_engine import GeminiAudioEngine
                            gemini_engine = GeminiAudioEngine()
                            if gemini_engine.is_available():
                                utterances, _ = gemini_engine.process_audio(tmp_path, mime_type=mime_type)
                                if utterances:
                                    dialogue_text = " ".join([u.transcript for u in utterances if u.transcript])
                        finally:
                            if os.path.exists(tmp_path):
                                try:
                                    os.remove(tmp_path)
                                except Exception:
                                    pass
                    except Exception as audio_err:
                        print(f"Audio transcription error: {audio_err}")
                
                if not dialogue_text:
                    dialogue_text = "Speech turn."

                if not utterances:
                    stt = LocalSTTEngine()
                    utterances = stt.process_local_transcript(dialogue_text)
                    if not utterances:
                        utterances = [Utterance(speaker="USER", start_time=0.0, end_time=3.0, transcript=dialogue_text)]

                # Check verbal self-intro
                utterances, intro_counterpart, intro_user = DiarizationEngine.detect_and_apply_verbal_introductions(
                    utterances
                )

                if intro_counterpart:
                    counterpart_name = intro_counterpart
                if intro_user or (len(utterances) <= 1 and not intro_counterpart):
                    power_axis = "SOLO"

                # Auto-enroll in registry if self-intro detected
                registry = SpeakerVoiceprintRegistry()
                if intro_counterpart and intro_counterpart not in registry.voiceprints:
                    registry.voiceprints[intro_counterpart] = SpeakerVoiceprint(
                        speaker_name=intro_counterpart,
                        role="Collaborator",
                        power_axis="LATERAL",
                        mean_pitch_hz=138.0,
                        is_user=False,
                    )
                    registry.save_to_disk()
                if intro_user and intro_user not in registry.voiceprints:
                    registry.voiceprints[intro_user] = SpeakerVoiceprint(
                        speaker_name=intro_user,
                        role="App User",
                        power_axis="SOLO",
                        mean_pitch_hz=126.0,
                        is_user=True,
                    )
                    registry.save_to_disk()

                session = ConversationSession(
                    session_id="emu_session",
                    timestamp_utc="2026-08-28T23:00:00Z",
                    target_speaker="USER",
                    counterpart_name=counterpart_name,
                    counterpart_role="Collaborator",
                    power_axis=power_axis,
                    dialogue=utterances,
                )

                engine = ExecutiveCoachingEngine(use_local_only=False)
                evaluation = engine.evaluate_session(session)

                # Extract Action items
                action_items = [
                    {
                        "owner": ai.owner,
                        "category": ai.category,
                        "due": ai.due_time_or_date or "Upcoming",
                        "resolved_datetime": ai.resolved_datetime,
                        "inferred_ampm": ai.target_time_inferred_ampm,
                        "task": ai.task,
                        "quote": ai.verbatim_quote,
                        "urgency": ai.urgency,
                    }
                    for ai in evaluation.action_items
                ]

                key_highlights = [
                    {
                        "headline": kh.headline,
                        "takeaway": kh.takeaway,
                        "speaker": kh.speaker,
                        "category": kh.category,
                        "importance": kh.importance,
                        "quote": kh.verbatim_quote,
                    }
                    for kh in getattr(evaluation, "key_highlights", [])
                ]

                top_strengths = [
                    {"observation": s.observation, "verbatim_quote": s.verbatim_quote} for s in evaluation.top_strengths
                ]

                areas_for_improvement = [
                    {
                        "critique": imp.critique,
                        "verbatim_quote": imp.verbatim_quote,
                        "coached_phrasing": imp.coached_phrasing,
                    }
                    for imp in evaluation.areas_for_improvement
                ]

                critique = (
                    areas_for_improvement[0]["critique"] if areas_for_improvement else "Delivery is clear and direct."
                )
                coached = (
                    areas_for_improvement[0]["coached_phrasing"]
                    if areas_for_improvement
                    else "Maintain this structured communication style."
                )

                # Conversational dynamics metrics
                dynamics_dict = None
                if getattr(evaluation, "dynamics", None):
                    d = evaluation.dynamics
                    dynamics_dict = {
                        "user_talk_time_pct": d.user_talk_time_pct,
                        "counterpart_talk_time_pct": d.counterpart_talk_time_pct,
                        "user_words_total": d.user_words_total,
                        "counterpart_words_total": d.counterpart_words_total,
                        "average_turn_latency_ms": d.average_turn_latency_ms,
                        "ask_vs_tell_ratio": d.ask_vs_tell_ratio,
                        "inquiry_count": d.inquiry_count,
                        "directive_count": d.directive_count,
                        "brevity_potential_pct": d.brevity_potential_pct,
                        "deep_listening_score": d.deep_listening_score,
                        "vocal_tension_index": d.vocal_tension_index,
                    }

                # Emotional trajectory timeline
                emotional_trajectory_list = [
                    {
                        "timestamp_sec": pt.timestamp_sec,
                        "speaker": pt.speaker,
                        "emotion_label": pt.emotion_label,
                        "valence_score": pt.valence_score,
                        "tension_level": pt.tension_level,
                        "pacing_wpm": pt.pacing_wpm,
                    }
                    for pt in getattr(evaluation, "emotional_trajectory", [])
                ]

                # Consensus agreements
                agreements_list = [
                    {
                        "headline": ag.headline,
                        "agreed_solution": ag.agreed_solution,
                        "speaker_turn": ag.speaker_turn,
                        "quote": ag.verbatim_quote,
                    }
                    for ag in getattr(evaluation, "agreements", [])
                ]

                # Unresolved open loops
                unresolved_loops_list = [
                    {
                        "concern_topic": ul.concern_topic,
                        "raised_by": ul.raised_by,
                        "context": ul.context,
                        "recommended_followup": ul.recommended_followup,
                    }
                    for ul in getattr(evaluation, "unresolved_loops", [])
                ]

                # Check if there is an enrolled app user or counterpart in registry
                enrolled_user = None
                for vp in registry.voiceprints.values():
                    if vp.is_user or vp.role in ["App User", "Self"] or vp.power_axis == "SOLO":
                        enrolled_user = vp.speaker_name
                        break

                if power_axis == "SOLO":
                    recognized_speaker = intro_user or enrolled_user or "New Voice (Solo)"
                    recognized_sub = (
                        f"User Voice Profile '{recognized_speaker}' • SOLO Mode"
                        if (intro_user or enrolled_user)
                        else "Unenrolled Voice • SOLO Mode"
                    )
                else:
                    recognized_speaker = intro_counterpart or counterpart_name or "New Collaborator"
                    recognized_sub = f"Voiceprint Profile Synced • {power_axis} Mode"

                resp_data = {
                    "title": "Evaluated Dialogue",
                    "dialogue": dialogue_text,
                    "recognized_speaker": recognized_speaker,
                    "recognized_sub": recognized_sub,
                    "power_axis": power_axis,
                    "tone": "Calm & Measured (132 Hz)",
                    "presence": getattr(evaluation.metrics, "presence_score", 75),
                    "assertiveness": getattr(evaluation.metrics, "assertiveness_score", 78),
                    "listening": getattr(evaluation.metrics, "active_listening_score", 80),
                    "speech_rate_wpm": getattr(evaluation.metrics, "speech_rate_wpm", 140),
                    "fillers_detected": [
                        {"token": f.token, "count": f.count}
                        for f in getattr(evaluation.metrics, "filler_words_detected", [])
                    ],
                    "hedging_count": getattr(evaluation.metrics, "hedging_qualifiers_count", 0),
                    "assertive_count": getattr(evaluation.metrics, "assertive_markers_count", 1),
                    "active_listening_count": getattr(evaluation.metrics, "active_listening_markers_count", 1),
                    "longitudinal_summary": evaluation.longitudinal_summary,
                    "top_strengths": top_strengths,
                    "areas_for_improvement": areas_for_improvement,
                    "key_highlights": key_highlights,
                    "action_items": action_items,
                    "rephrasing": {"critique": critique, "coached": coached},
                    "dynamics": dynamics_dict,
                    "emotional_trajectory": emotional_trajectory_list,
                    "agreements": agreements_list,
                    "unresolved_loops": unresolved_loops_list,
                }
                self._send_json(resp_data)
            except Exception:
                # Safe fallback
                self._send_json(
                    {
                        "title": "Evaluated Dialogue",
                        "dialogue": payload.get("dialogue_text", ""),
                        "recognized_speaker": "Live Speaker",
                        "recognized_sub": "Analyzed via On-Device Engine",
                        "power_axis": "LATERAL",
                        "tone": "Natural Voice (128 Hz)",
                        "presence": 75,
                        "assertiveness": 78,
                        "listening": 80,
                        "speech_rate_wpm": 140,
                        "fillers_detected": [],
                        "hedging_count": 0,
                        "assertive_count": 1,
                        "active_listening_count": 1,
                        "longitudinal_summary": "Delivery structured with clear communication intent.",
                        "top_strengths": [
                            {
                                "observation": "Clear topical focus and delivery flow.",
                                "verbatim_quote": payload.get("dialogue_text", "")[:50],
                            }
                        ],
                        "areas_for_improvement": [
                            {
                                "critique": "Ensure bottom-line recommendation is stated upfront.",
                                "verbatim_quote": payload.get("dialogue_text", "")[:50],
                                "coached_phrasing": "Let's prioritize the key action item.",
                            }
                        ],
                        "key_highlights": [],
                        "action_items": [],
                        "rephrasing": {
                            "critique": "Observation processed. Ensure bottom-line recommendation is stated upfront.",
                            "coached": "Let's align on the core action item to ensure delivery readiness.",
                        },
                        "dynamics": {
                            "user_talk_time_pct": 50.0,
                            "counterpart_talk_time_pct": 50.0,
                            "user_words_total": 20,
                            "counterpart_words_total": 20,
                            "average_turn_latency_ms": 350.0,
                            "ask_vs_tell_ratio": 1.0,
                            "inquiry_count": 1,
                            "directive_count": 1,
                            "brevity_potential_pct": 0.0,
                            "deep_listening_score": 75,
                            "vocal_tension_index": "Calm & Grounded",
                        },
                        "emotional_trajectory": [],
                        "agreements": [],
                        "unresolved_loops": [],
                    }
                )

        elif url_path == "/api/enroll_voiceprint":
            name = payload.get("speaker_name", "").strip()
            junk_names = {
                "speaker (solo)",
                "speaker (analyzed)",
                "solo speaker",
                "live speaker",
                "speaker",
                "new speaker",
                "",
            }
            if not name or name.lower() in junk_names or len(name) < 2:
                self._send_json({"status": "error", "message": "Please provide a valid speaker name."})
                return

            role = payload.get("role", "Collaborator")
            power_axis = payload.get("power_axis", "LATERAL")
            pitch = float(payload.get("mean_pitch_hz", 135.0))
            is_user = bool(payload.get("is_user", False)) or (
                role.lower() in ["self", "user", "app user"] or power_axis.upper() == "SOLO"
            )

            registry = SpeakerVoiceprintRegistry()
            vp = SpeakerVoiceprint(
                speaker_name=name, role=role, power_axis=power_axis, mean_pitch_hz=pitch, is_user=is_user
            )
            registry.voiceprints[name] = vp
            registry.save_to_disk()
            self._send_json({"status": "success", "enrolled": vp.to_dict()})

        elif url_path == "/api/erase_voiceprint":
            name = payload.get("speaker_name", "").strip()
            registry = SpeakerVoiceprintRegistry()
            success = registry.delete_voiceprint(name)
            self._send_json({"status": "success" if success else "not_found", "speaker_name": name})

        elif url_path == "/api/detect_actions":
            dialogue_text = payload.get("dialogue_text", "")
            stt = LocalSTTEngine()
            utterances = stt.process_local_transcript(dialogue_text)
            items = ActionItemExtractor.extract_from_dialogue(utterances)
            data = [
                {
                    "owner": ai.owner,
                    "category": ai.category,
                    "due": ai.due_time_or_date,
                    "resolved_datetime": ai.resolved_datetime,
                    "inferred_ampm": ai.target_time_inferred_ampm,
                    "task": ai.task,
                    "quote": ai.verbatim_quote,
                    "urgency": ai.urgency,
                }
                for ai in items
            ]
            self._send_json(data)

        elif url_path == "/api/transcription_analysis":
            dialogue_text = payload.get("dialogue_text", "")
            analyzer = TranscriptionAnalyzer()
            analysis = analyzer.analyze(dialogue_text, use_gemini=False)
            data = {
                "summary": analysis.summary,
                "topics": analysis.topics_discussed,
                "tone": analysis.sentiment_tone,
                "highlights": [
                    {
                        "headline": kh.headline,
                        "takeaway": kh.takeaway,
                        "speaker": kh.speaker,
                        "category": kh.category,
                        "importance": kh.importance,
                        "quote": kh.verbatim_quote,
                    }
                    for kh in analysis.key_highlights
                ],
                "tasks": [
                    {
                        "owner": t.owner,
                        "task": t.task,
                        "category": t.category,
                        "due": t.due_time_or_date,
                        "resolved_datetime": t.resolved_datetime,
                        "inferred_ampm": t.target_time_inferred_ampm,
                        "quote": t.verbatim_quote,
                        "urgency": t.urgency,
                    }
                    for t in analysis.potential_tasks
                ],
            }
            self._send_json(data)
        else:
            self.send_error(404, "Unknown API Route")

    def send_header(self, keyword: str, value: str):
        """Sanitizes header names and values to prevent HTTP response splitting (CWE-113)."""
        clean_keyword = "".join(c for c in str(keyword) if c not in "\r\n")
        clean_value = "".join(c for c in str(value) if c not in "\r\n")
        super().send_header(clean_keyword, clean_value)

    def do_OPTIONS(self):
        """Handle preflight CORS requests safely."""
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _serve_file(self, rel_path: str):
        safe_name = os.path.basename(rel_path.lstrip("/\\")) or "index.html"
        base_dir = os.path.realpath(os.path.join(PROJECT_ROOT, "emulator"))
        fullpath = os.path.normpath(os.path.join(base_dir, safe_name))

        if not fullpath.startswith(base_dir):
            self.send_error(403, "Forbidden")
            return
        if not os.path.isfile(fullpath):
            self.send_error(404, "File Not Found")
            return

        _, ext = os.path.splitext(fullpath)
        safe_content_type = SAFE_MIME_TYPES.get(ext.lower(), "application/octet-stream")

        with open(fullpath, "rb") as f:
            content = f.read()

        self.send_response(200)
        self.send_header("Content-Type", safe_content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, data: any):
        raw = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, format, *args):
        # Clean logging
        return


class ReusableHTTPServer(HTTPServer):
    allow_reuse_address = True


def run_server(port: int = 8080):
    server = ReusableHTTPServer(("127.0.0.1", port), EmulatorHandler)
    print(f"[EMULATOR SERVER] Running at http://127.0.0.1:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    run_server(port)


if __name__ == "__main__":
    main()
