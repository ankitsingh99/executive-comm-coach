"""
Sarvam AI Saaras Client for Bilingual Hinglish Speech Recognition and Speaker Diarization.
Handles multi-speaker Indic / English code-mixed audio transcription with full speaker attribution.
"""

import os
import json
import uuid
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional

try:
    from ..engine.schema import Utterance
except (ImportError, ValueError):
    from engine.schema import Utterance


class SarvamSpeechClient:
    """
    Client for Sarvam AI Saaras bilingual Hinglish ASR & Speaker Diarization.
    Extracts verbatim transcripts and assigns distinct speaker tags to each utterance turn.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("SARVAM_API_KEY", "")
        self.base_url = "https://api.sarvam.ai/speech-to-text"

    def is_available(self) -> bool:
        """Returns True if Sarvam API key is configured."""
        return bool(self.api_key and self.api_key.strip())

    def transcribe_audio_chunk(
        self,
        audio_file_path: str,
        language_code: str = "hi-IN",
        model: str = "saaras:v2",
        with_diarization: bool = True
    ) -> List[Utterance]:
        """
        Submits audio file to Sarvam AI Saaras endpoint with speaker diarization enabled.
        Returns a list of speaker-tagged Utterance objects with timestamps.
        """
        if not self.is_available() or not os.path.exists(audio_file_path):
            return []

        try:
            with open(audio_file_path, "rb") as f:
                file_bytes = f.read()

            filename = os.path.basename(audio_file_path)
            content_type, body = self._build_multipart_payload(
                fields={
                    "model": model,
                    "language_code": language_code,
                    "with_diarization": "true" if with_diarization else "false",
                    "with_timestamps": "true"
                },
                file_field="file",
                filename=filename,
                file_bytes=file_bytes
            )

            req = urllib.request.Request(
                self.base_url,
                data=body,
                headers={
                    "api-subscription-key": self.api_key,
                    "Content-Type": content_type
                },
                method="POST"
            )

            with urllib.request.urlopen(req, timeout=30.0) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode("utf-8"))
                    return self._parse_sarvam_response(data)
                return []
        except Exception:
            return []

    def _build_multipart_payload(
        self,
        fields: Dict[str, str],
        file_field: str,
        filename: str,
        file_bytes: bytes
    ) -> (str, bytes):
        """Constructs multipart/form-data payload without external dependencies."""
        boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
        body = bytearray()

        for key, val in fields.items():
            body.extend(f"--{boundary}\r\n".encode("utf-8"))
            body.extend(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8"))
            body.extend(f"{val}\r\n".encode("utf-8"))

        mime = "audio/wav" if filename.lower().endswith(".wav") else "application/octet-stream"
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'.encode("utf-8"))
        body.extend(f"Content-Type: {mime}\r\n\r\n".encode("utf-8"))
        body.extend(file_bytes)
        body.extend(b"\r\n")

        body.extend(f"--{boundary}--\r\n".encode("utf-8"))
        content_type = f"multipart/form-data; boundary={boundary}"
        return content_type, bytes(body)

    def _parse_sarvam_response(self, response_json: Dict[str, Any]) -> List[Utterance]:
        """
        Parses Sarvam API response into speaker-attributed Utterances.
        Maps raw speaker IDs ('speaker_0', 'speaker_1', etc.) to semantic roles:
        'USER' for speaker_0, 'COUNTERPART' for speaker_1, or preserves distinct speaker tags.
        """
        utterances: List[Utterance] = []
        diarized_entries = response_json.get("diarized_transcript", {}).get("entries", [])

        if diarized_entries:
            speaker_map = {}
            for entry in diarized_entries:
                raw_spk = entry.get("speaker_id", "speaker_0").strip()
                if raw_spk not in speaker_map:
                    if len(speaker_map) == 0:
                        speaker_map[raw_spk] = "USER"
                    elif len(speaker_map) == 1:
                        speaker_map[raw_spk] = "COUNTERPART"
                    else:
                        speaker_map[raw_spk] = f"SPEAKER_{len(speaker_map)+1:02d}"

                speaker_label = speaker_map[raw_spk]
                text = entry.get("transcript", "").strip()
                start_t = float(entry.get("start_time_seconds", 0.0))
                end_t = float(entry.get("end_time_seconds", start_t + 1.0))

                if text:
                    utterances.append(
                        Utterance(
                            speaker=speaker_label,
                            start_time=round(start_t, 2),
                            end_time=round(end_t, 2),
                            transcript=text
                        )
                    )
            if utterances:
                return utterances

        # Fallback: if diarization entries are empty, parse top-level transcript
        raw_transcript = response_json.get("transcript", "").strip()
        if raw_transcript:
            utterances.append(
                Utterance(
                    speaker="USER",
                    start_time=0.0,
                    end_time=5.0,
                    transcript=raw_transcript
                )
            )

        return utterances
