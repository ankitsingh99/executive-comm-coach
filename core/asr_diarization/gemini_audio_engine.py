"""
Gemini Multimodal Audio Processing & Transcription Engine.
Leverages Gemini 2.5 Flash native audio understanding for high-accuracy speech-to-text,
diarization, phonetic hesitation preservation, and vocal tone analysis.
"""

import os
import json
import logging
import warnings
from typing import List, Tuple, Optional, Dict, Any

# Suppress GenAI automatic function calling warning
logging.getLogger("google.genai").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*automatic function calling.*")

try:
    from ..engine.schema import Utterance, SpeakerAcousticProfile, AcousticAnalysisResult
    from ..config import get_gemini_api_key, GEMINI_MODEL
except (ImportError, ValueError):
    from engine.schema import Utterance, SpeakerAcousticProfile, AcousticAnalysisResult
    from config import get_gemini_api_key, GEMINI_MODEL


class GeminiAudioEngine:
    """
    Multimodal audio processing engine powered by Google Gemini.
    Transcribes audio verbatim, detects speakers, and classifies vocal delivery directly from acoustics.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = GEMINI_MODEL):
        self.api_key = api_key or get_gemini_api_key()
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None and self.api_key:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except Exception as e:
                self._client = None
        return self._client

    def is_available(self) -> bool:
        """Checks if Gemini API key is configured and client is initialized."""
        return bool(self.api_key and self._get_client() is not None)

    def process_audio(
        self,
        audio_wav_path: str,
        speaker_id: str = "USER"
    ) -> Tuple[List[Utterance], AcousticAnalysisResult]:
        """
        Transcribes audio and extracts acoustic voice/tone characteristics using Gemini.
        """
        if not os.path.exists(audio_wav_path) or not self.is_available():
            return [], AcousticAnalysisResult()

        client = self._get_client()
        if not client:
            return [], AcousticAnalysisResult()

        try:
            from google.genai import types

            with open(audio_wav_path, "rb") as f:
                audio_bytes = f.read()

            if len(audio_bytes) < 1000:
                return [], AcousticAnalysisResult()

            audio_part = types.Part.from_bytes(
                data=audio_bytes,
                mime_type="audio/wav"
            )

            prompt = """Analyze this audio recording with high precision for speech-to-text, speaker diarization, and acoustic tone.

Return pure JSON with the following structure:
{
  "transcription": [
    {
      "speaker": "USER",
      "start_time": 0.0,
      "end_time": 3.5,
      "transcript": "Exact verbatim spoken words here, including fillers like um, ah, basically, etc."
    },
    {
      "speaker": "COUNTERPART",
      "start_time": 3.6,
      "end_time": 7.0,
      "transcript": "Exact verbatim reply from the second speaker."
    }
  ],
  "speaker_count": 2,
  "overall_tone": "Calm & Measured",
  "speakers": [
    {
      "speaker_id": "USER",
      "tone_label": "Calm & Measured",
      "pitch_hz": 150.0,
      "talk_time_percentage": 60.0,
      "confidence_score": 0.95
    },
    {
      "speaker_id": "COUNTERPART",
      "tone_label": "Assertive & Decisive",
      "pitch_hz": 180.0,
      "talk_time_percentage": 40.0,
      "confidence_score": 0.92
    }
  ]
}

Instructions:
1. Capture every spoken word VERBATIM. Do NOT omit filler vocalizations (um, umm, uh, hmm, aaah, matlab, yaani, etc.).
2. SPEAKER DIARIZATION (MANDATORY):
   - You MUST accurately tag which person said what for every single utterance.
   - If only 1 person speaks in the audio, label their speaker as "USER".
   - If multiple distinct voices/people speak:
     * Label the main speaker (or first speaker) as "USER".
     * Label other interlocutors as "COUNTERPART" (or "SPEAKER_02", "SPEAKER_03" if 3+ people).
     * Split every change in speaker into a separate turn in "transcription".
   - NEVER combine different speakers' speech into one utterance.
3. For each detected speaker, provide their tone_label, estimated pitch_hz, and talk_time_percentage.
4. Return ONLY valid JSON without markdown wrapping.
"""

            config_kwargs = {"response_mime_type": "application/json"}
            try:
                config_kwargs["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(disable=True)
            except Exception:
                pass

            response = client.models.generate_content(
                model=self.model,
                contents=[audio_part, prompt],
                config=types.GenerateContentConfig(**config_kwargs)
            )

            raw_text = response.text.strip()
            if raw_text.startswith("```"):
                raw_text = raw_text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            data = json.loads(raw_text)

            # Parse Utterances with speaker normalization
            utterances: List[Utterance] = []
            for item in data.get("transcription", []):
                raw_spk = str(item.get("speaker", speaker_id)).strip()
                # Normalize speaker tags
                if raw_spk.upper() in ["USER", "SPEAKER_0", "SPEAKER_01", "SPEAKER 1", "SELF"]:
                    spk = "USER"
                elif raw_spk.upper() in ["COUNTERPART", "SPEAKER_1", "SPEAKER_02", "SPEAKER 2", "OTHER"]:
                    spk = "COUNTERPART"
                else:
                    spk = raw_spk.upper()

                start = float(item.get("start_time", 0.0))
                end = float(item.get("end_time", 0.0))
                text = item.get("transcript", "").strip()
                if text:
                    utterances.append(Utterance(speaker=spk, start_time=start, end_time=end, transcript=text))

            # Parse Acoustic & Tone profiles
            spk_count = int(data.get("speaker_count", max(1, len(data.get("speakers", [])))))
            overall_tone = data.get("overall_tone", "Natural & Conversational")

            speaker_profiles: List[SpeakerAcousticProfile] = []
            for s in data.get("speakers", []):
                speaker_profiles.append(
                    SpeakerAcousticProfile(
                        speaker_id=s.get("speaker_id", "SPEAKER_01"),
                        mean_pitch_hz=float(s.get("pitch_hz", 150.0)),
                        pitch_range_hz=35.0,
                        energy_rms=0.05,
                        speech_rate_wpm=140.0,
                        tone_label=s.get("tone_label", overall_tone),
                        talk_time_percentage=float(s.get("talk_time_percentage", 100.0)),
                        confidence_score=float(s.get("confidence_score", 0.95))
                    )
                )

            if not speaker_profiles:
                speaker_profiles = [
                    SpeakerAcousticProfile(
                        speaker_id="SPEAKER_01",
                        tone_label=overall_tone,
                        talk_time_percentage=100.0,
                        confidence_score=0.95
                    )
                ]

            acoustic_res = AcousticAnalysisResult(
                detected_speaker_count=spk_count,
                is_multi_speaker=(spk_count > 1),
                speakers=speaker_profiles,
                overall_tone=overall_tone,
                turn_taking_events=max(0, spk_count - 1)
            )

            return utterances, acoustic_res

        except Exception as e:
            return [], AcousticAnalysisResult()
