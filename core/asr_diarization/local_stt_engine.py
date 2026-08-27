"""
Local On-Device Speech Recognition & Diarization Engine.
Powered by NVIDIA Parakeet STT as primary acoustic model with Faster-Whisper fallback.
"""

import os
import re
from typing import List, Dict, Any, Optional

try:
    from ..engine.schema import Utterance
    from .nvidia_parakeet_engine import NvidiaParakeetEngine
except (ImportError, ValueError):
    from engine.schema import Utterance
    from asr_diarization.nvidia_parakeet_engine import NvidiaParakeetEngine


class LocalSTTEngine:
    """
    On-device speech recognizer and speaker diarizer.
    Utilizes NVIDIA Parakeet CTC as primary SOTA model for transcription.
    """

    def __init__(self, sample_rate: int = 16000, use_parakeet: bool = True):
        self.sample_rate = sample_rate
        self.use_parakeet = use_parakeet
        self._parakeet_engine = None
        self._whisper_model = None

    def _get_parakeet_engine(self):
        if self._parakeet_engine is None and self.use_parakeet:
            try:
                self._parakeet_engine = NvidiaParakeetEngine()
            except Exception:
                self._parakeet_engine = None
        return self._parakeet_engine

    def _get_whisper_model(self):
        if self._whisper_model is None:
            try:
                from faster_whisper import WhisperModel
                self._whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
            except Exception:
                self._whisper_model = None
        return self._whisper_model

    def transcribe_audio_file(self, audio_wav_path: str, speaker_id: str = "USER") -> List[Utterance]:
        """
        Transcribes a recorded WAV audio file using NVIDIA Parakeet (or Whisper fallback).
        """
        if not os.path.exists(audio_wav_path):
            return []

        # 1. Primary Engine: NVIDIA Parakeet
        parakeet = self._get_parakeet_engine()
        if parakeet is not None:
            try:
                results = parakeet.transcribe_audio_file(audio_wav_path, speaker_id=speaker_id)
                if results and results[0].transcript.strip():
                    return results
            except Exception:
                pass

        # 2. Fallback Engine: Faster-Whisper
        whisper_model = self._get_whisper_model()
        if whisper_model is not None:
            try:
                segments, info = whisper_model.transcribe(audio_wav_path, beam_size=3)
                utterances = []
                for seg in segments:
                    text = seg.text.strip()
                    if text:
                        utterances.append(
                            Utterance(
                                speaker=speaker_id,
                                start_time=round(seg.start, 2),
                                end_time=round(seg.end, 2),
                                transcript=text
                            )
                        )
                if utterances:
                    return utterances
            except Exception:
                pass

        return []

    def process_local_transcript(
        self,
        raw_text: str,
        user_speaker_id: str = "USER",
        counterpart_speaker_id: str = "COUNTERPART"
    ) -> List[Utterance]:
        """
        Parses multi-line script format into timestamped Utterances.
        """
        utterances: List[Utterance] = []
        lines = [line.strip() for line in raw_text.strip().split("\n") if line.strip()]
        current_time = 0.0

        for line in lines:
            match = re.match(r"^([A-Za-z0-9_\s]+?):\s*(.+)$", line)
            if match:
                speaker_raw, text = match.groups()
                speaker_clean = speaker_raw.strip().upper()
                speaker = "USER" if "USER" in speaker_clean else "COUNTERPART"
            else:
                speaker = "USER" if len(utterances) % 2 == 0 else "COUNTERPART"
                text = line

            words = len(text.split())
            duration = max(1.5, round(words / 2.5, 1))
            end_time = round(current_time + duration, 1)

            utterances.append(
                Utterance(
                    speaker=speaker,
                    start_time=current_time,
                    end_time=end_time,
                    transcript=text.strip()
                )
            )
            current_time = round(end_time + 0.4, 1)

        return utterances
