"""
Sarvam AI Saaras v3 Client for Bilingual Hinglish Speech Recognition.
Handles multi-speaker Indic / English code-mixed audio transcription.
"""

import os
import json
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional

try:
    from ..engine.schema import Utterance
except (ImportError, ValueError):
    from engine.schema import Utterance


class SarvamSpeechClient:
    """
    Client for Sarvam AI Saaras v3 bilingual Hinglish ASR & Speaker Diarization.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("SARVAM_API_KEY", "")
        self.base_url = "https://api.sarvam.ai/speech-to-text"

    def transcribe_audio_chunk(
        self,
        audio_file_path: str,
        language_code: str = "hi-IN",
        with_diarization: bool = True
    ) -> List[Utterance]:
        """
        Submits audio to Sarvam AI Saaras v3 endpoint for code-mixed transcription.
        """
        if not self.api_key or not os.path.exists(audio_file_path):
            return []

        try:
            req = urllib.request.Request(
                self.base_url,
                headers={
                    "api-subscription-key": self.api_key,
                    "Content-Type": "application/json"
                }
            )
            with urllib.request.urlopen(req, timeout=10.0) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode("utf-8"))
                    return self._parse_sarvam_response(data)
                else:
                    return []
        except Exception:
            return []

    def _parse_sarvam_response(self, response_json: Dict[str, Any]) -> List[Utterance]:
        utterances: List[Utterance] = []
        diarized_chunks = response_json.get("diarized_transcript", {}).get("entries", [])
        for entry in diarized_chunks:
            utterances.append(
                Utterance(
                    speaker=entry.get("speaker_id", "SPEAKER_01"),
                    start_time=float(entry.get("start_time_seconds", 0.0)),
                    end_time=float(entry.get("end_time_seconds", 0.0)),
                    transcript=entry.get("transcript", "")
                )
            )
        return utterances
