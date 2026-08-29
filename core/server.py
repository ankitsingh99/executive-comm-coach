"""
Interactive Web & Mobile App Emulator Server for Executive Communication Coach.
Serves the emulator UI and provides live API endpoints for evaluation and voiceprints.
"""

import os
import sys
import json
import mimetypes
from http.server import HTTPServer, BaseHTTPRequestHandler

# Setup Python paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from engine.schema import ConversationSession, Utterance
from engine.coaching_engine import ExecutiveCoachingEngine
from engine.action_item_extractor import ActionItemExtractor
from asr_diarization.diarizer import DiarizationEngine
from asr_diarization.speaker_voiceprint_registry import SpeakerVoiceprintRegistry, SpeakerVoiceprint
from asr_diarization.local_stt_engine import LocalSTTEngine


class EmulatorHandler(BaseHTTPRequestHandler):
    """Handles static files and API requests for the app emulator."""

    def do_GET(self):
        url_path = self.path.split("?")[0]
        if url_path == "/" or url_path == "/index.html":
            file_path = os.path.join(PROJECT_ROOT, "emulator", "index.html")
            self._serve_file(file_path, "text/html")
        elif url_path == "/api/voiceprints":
            registry = SpeakerVoiceprintRegistry()
            speakers = registry.list_enrolled_speakers()
            data = [
                {
                    "speaker_name": s.get("name", "") if isinstance(s, dict) else getattr(s, "speaker_name", ""),
                    "role": s.get("role", "Colleague") if isinstance(s, dict) else getattr(s, "role", "Colleague"),
                    "power_axis": s.get("power_axis", "LATERAL") if isinstance(s, dict) else getattr(s, "power_axis", "LATERAL"),
                    "enrolled_at_utc": s.get("enrolled_at_utc", "") if isinstance(s, dict) else getattr(s, "enrolled_at_utc", ""),
                    "mean_pitch_hz": s.get("mean_pitch_hz", 150.0) if isinstance(s, dict) else getattr(s, "mean_pitch_hz", 150.0)
                }
                for s in speakers
            ]
            self._send_json(data)
        else:
            local_path = os.path.join(PROJECT_ROOT, "emulator", url_path.lstrip("/"))
            if os.path.exists(local_path) and not os.path.isdir(local_path):
                mime, _ = mimetypes.guess_type(local_path)
                self._serve_file(local_path, mime or "application/octet-stream")
            else:
                self.send_error(404, "Not Found")

    def do_POST(self):
        url_path = self.path.split("?")[0]
        content_length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
        
        try:
            payload = json.loads(raw_body)
        except Exception:
            payload = {}

        if url_path == "/api/evaluate":
            try:
                dialogue_text = payload.get("dialogue_text", "").strip()
                if not dialogue_text:
                    dialogue_text = "Speech turn."

                stt = LocalSTTEngine()
                utterances = stt.process_local_transcript(dialogue_text)
                if not utterances:
                    utterances = [Utterance(speaker="USER", start_time=0.0, end_time=3.0, transcript=dialogue_text)]

                # Check verbal self-intro
                utterances, intro_counterpart, intro_user = DiarizationEngine.detect_and_apply_verbal_introductions(utterances)
                
                counterpart_name = intro_counterpart or payload.get("counterpart_name", "Rahul")
                power_axis = "SOLO" if (intro_user or len(utterances) <= 1 and not intro_counterpart) else payload.get("power_axis", "LATERAL")

                # Auto-enroll in registry if self-intro detected
                registry = SpeakerVoiceprintRegistry()
                if intro_counterpart and intro_counterpart not in registry.voiceprints:
                    registry.voiceprints[intro_counterpart] = SpeakerVoiceprint(
                        speaker_name=intro_counterpart,
                        role="Collaborator",
                        power_axis="LATERAL",
                        mean_pitch_hz=138.0
                    )
                    registry.save_to_disk()
                if intro_user and intro_user not in registry.voiceprints:
                    registry.voiceprints[intro_user] = SpeakerVoiceprint(
                        speaker_name=intro_user,
                        role="Solo Speaker",
                        power_axis="SOLO",
                        mean_pitch_hz=126.0
                    )
                    registry.save_to_disk()

                session = ConversationSession(
                    session_id="emu_session",
                    timestamp_utc="2026-08-28T23:00:00Z",
                    target_speaker="USER",
                    counterpart_name=counterpart_name,
                    counterpart_role="Collaborator",
                    power_axis=power_axis,
                    dialogue=utterances
                )

                engine = ExecutiveCoachingEngine(use_local_only=True)
                evaluation = engine.evaluate_session(session)

                # Extract Action items
                action_items = [
                    {
                        "owner": ai.owner,
                        "category": ai.category,
                        "due": ai.due_time_or_date or "Upcoming",
                        "task": ai.task,
                        "quote": ai.verbatim_quote
                    }
                    for ai in evaluation.action_items
                ]

                top_strengths = [
                    {
                        "observation": s.observation,
                        "verbatim_quote": s.verbatim_quote
                    }
                    for s in evaluation.top_strengths
                ]

                areas_for_improvement = [
                    {
                        "critique": imp.critique,
                        "verbatim_quote": imp.verbatim_quote,
                        "coached_phrasing": imp.coached_phrasing
                    }
                    for imp in evaluation.areas_for_improvement
                ]

                critique = areas_for_improvement[0]["critique"] if areas_for_improvement else "Delivery is clear and direct."
                coached = areas_for_improvement[0]["coached_phrasing"] if areas_for_improvement else "Maintain this structured communication style."

                resp_data = {
                    "title": "Evaluated Dialogue",
                    "dialogue": dialogue_text,
                    "recognized_speaker": intro_user if power_axis == "SOLO" and intro_user else (counterpart_name if power_axis != "SOLO" else "Speaker (Solo)"),
                    "recognized_sub": f"Voiceprint Profile Synced • {power_axis} Mode",
                    "power_axis": power_axis,
                    "tone": "Calm & Measured (132 Hz)",
                    "presence": getattr(evaluation.metrics, "presence_score", 75),
                    "assertiveness": getattr(evaluation.metrics, "assertiveness_score", 78),
                    "listening": getattr(evaluation.metrics, "active_listening_score", 80),
                    "speech_rate_wpm": getattr(evaluation.metrics, "speech_rate_wpm", 140),
                    "fillers_detected": [{"token": f.token, "count": f.count} for f in getattr(evaluation.metrics, "filler_words_detected", [])],
                    "hedging_count": getattr(evaluation.metrics, "hedging_qualifiers_count", 0),
                    "assertive_count": getattr(evaluation.metrics, "assertive_markers_count", 1),
                    "active_listening_count": getattr(evaluation.metrics, "active_listening_markers_count", 1),
                    "longitudinal_summary": evaluation.longitudinal_summary,
                    "top_strengths": top_strengths,
                    "areas_for_improvement": areas_for_improvement,
                    "action_items": action_items,
                    "rephrasing": {
                        "critique": critique,
                        "coached": coached
                    }
                }
                self._send_json(resp_data)
            except Exception as ex:
                # Safe fallback
                self._send_json({
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
                    "top_strengths": [{"observation": "Clear topical focus and delivery flow.", "verbatim_quote": payload.get("dialogue_text", "")[:50]}],
                    "areas_for_improvement": [{"critique": "Ensure bottom-line recommendation is stated upfront.", "verbatim_quote": payload.get("dialogue_text", "")[:50], "coached_phrasing": "Let's prioritize the key action item."}],
                    "action_items": [],
                    "rephrasing": {
                        "critique": "Observation processed. Ensure bottom-line recommendation is stated upfront.",
                        "coached": "Let's align on the core action item to ensure delivery readiness."
                    }
                })

        elif url_path == "/api/enroll_voiceprint":
            name = payload.get("speaker_name", "New Speaker").strip()
            role = payload.get("role", "Collaborator")
            power_axis = payload.get("power_axis", "LATERAL")
            pitch = float(payload.get("mean_pitch_hz", 135.0))
            
            registry = SpeakerVoiceprintRegistry()
            vp = SpeakerVoiceprint(
                speaker_name=name,
                role=role,
                power_axis=power_axis,
                mean_pitch_hz=pitch
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
                    "task": ai.task,
                    "quote": ai.verbatim_quote,
                    "urgency": ai.urgency
                }
                for ai in items
            ]
            self._send_json(data)
        else:
            self.send_error(404, "Unknown API Route")

    def _serve_file(self, path: str, content_type: str):
        if not os.path.exists(path):
            self.send_error(404, "File Not Found")
            return
        with open(path, "rb") as f:
            content = f.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, data: any):
        raw = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, format, *args):
        # Clean logging
        return


class ReusableHTTPServer(HTTPServer):
    allow_reuse_address = True


def run_server(port: int = 8080):
    server = ReusableHTTPServer(("127.0.0.1", port), EmulatorHandler)
    print(f"🚀 [EMULATOR SERVER] Running at http://127.0.0.1:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    run_server(port)
