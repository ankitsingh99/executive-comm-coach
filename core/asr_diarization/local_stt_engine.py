"""
Local On-Device Speech Recognition & Diarization Engine.
Powered by Google Gemini and Sarvam AI Saaras for Indic/Hinglish,
with NVIDIA Parakeet and Faster-Whisper local fallbacks.
"""

import os
import re
from typing import List

try:
    from ..engine.schema import Utterance
    from .nvidia_parakeet_engine import NvidiaParakeetEngine
    from .gemini_audio_engine import GeminiAudioEngine
    from .sarvam_client import SarvamSpeechClient
    from .indic_normalizer import IndicNormalizer
    from .acoustic_filler_detector import AcousticFillerDetector
except (ImportError, ValueError):
    from engine.schema import Utterance
    from asr_diarization.nvidia_parakeet_engine import NvidiaParakeetEngine
    from asr_diarization.gemini_audio_engine import GeminiAudioEngine
    from asr_diarization.sarvam_client import SarvamSpeechClient
    from asr_diarization.indic_normalizer import IndicNormalizer
    from asr_diarization.acoustic_filler_detector import AcousticFillerDetector


class LocalSTTEngine:
    """
    Speech recognizer and speaker diarizer.
    Utilizes Google Gemini and Sarvam AI as primary engines for bilingual Hinglish,
    with local NVIDIA Parakeet and Whisper fallbacks.
    """

    def __init__(self, sample_rate: int = 16000, use_parakeet: bool = True, model_size: str = "tiny"):
        self.sample_rate = sample_rate
        self.use_parakeet = use_parakeet
        self.model_size = model_size
        self._gemini_engine = GeminiAudioEngine()
        self._sarvam_client = SarvamSpeechClient()
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

                self._whisper_model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
            except Exception:
                self._whisper_model = None
        return self._whisper_model

    def transcribe_audio_file(self, audio_wav_path: str, speaker_id: str = "USER") -> List[Utterance]:
        """
        Transcribes a recorded WAV audio file using Gemini / Sarvam (or Parakeet / Whisper fallback).
        Applies Indic and Hinglish normalizations to ensure accurate downstream NLP.
        """
        if not os.path.exists(audio_wav_path):
            return []

        # 1. Primary Engine: Google Gemini (Highest multimodal accuracy & verbatim phonetic hesitations)
        if self._gemini_engine.is_available():
            try:
                gemini_utterances, _ = self._gemini_engine.process_audio(audio_wav_path, speaker_id=speaker_id)
                if gemini_utterances and any(u.transcript.strip() for u in gemini_utterances):
                    for u in gemini_utterances:
                        u.transcript = IndicNormalizer.normalize_text(u.transcript)
                    return gemini_utterances
            except Exception:
                pass

        # 2. Indic / Hinglish Specialized Engine: Sarvam AI Saaras
        if self._sarvam_client.is_available():
            try:
                sarvam_utterances = self._sarvam_client.transcribe_audio_chunk(
                    audio_wav_path, language_code="hi-IN", with_diarization=True
                )
                if sarvam_utterances and any(u.transcript.strip() for u in sarvam_utterances):
                    for u in sarvam_utterances:
                        u.transcript = IndicNormalizer.normalize_text(u.transcript)
                    return sarvam_utterances
            except Exception:
                pass

        # 3. Secondary Engine: NVIDIA Parakeet
        parakeet = self._get_parakeet_engine()
        if parakeet is not None:
            try:
                results = parakeet.transcribe_audio_file(audio_wav_path, speaker_id=speaker_id)
                if results and results[0].transcript.strip():
                    for u in results:
                        u.transcript = IndicNormalizer.normalize_text(u.transcript)
                    acoustic_fillers = AcousticFillerDetector().detect_fillers_from_wav(
                        audio_wav_path, speaker_segments=results
                    )
                    return AcousticFillerDetector.inject_fillers_into_utterances(results, acoustic_fillers)
            except Exception:
                pass

        # 4. Fallback Engine: Faster-Whisper with Bilingual Code-Mixed Prompt
        whisper_model = self._get_whisper_model()
        if whisper_model is not None:
            try:
                initial_prompt = "English, Hindi, and Hinglish dialogue. Transcribe code-mixed words verbatim like matlab, kal, deploy, sync."
                segments, info = whisper_model.transcribe(audio_wav_path, beam_size=3, initial_prompt=initial_prompt)
                utterances = []
                for seg in segments:
                    text = seg.text.strip()
                    if text:
                        norm_text = IndicNormalizer.normalize_text(text)
                        utterances.append(
                            Utterance(
                                speaker=speaker_id,
                                start_time=round(seg.start, 2),
                                end_time=round(seg.end, 2),
                                transcript=norm_text,
                            )
                        )
                if utterances:
                    acoustic_fillers = AcousticFillerDetector().detect_fillers_from_wav(
                        audio_wav_path, speaker_segments=utterances
                    )
                    return AcousticFillerDetector.inject_fillers_into_utterances(utterances, acoustic_fillers)
            except Exception:
                pass

        return []

    def process_local_transcript(
        self, raw_text: str, user_speaker_id: str = "USER", counterpart_speaker_id: str = "COUNTERPART"
    ) -> List[Utterance]:
        """
        Parses multi-line script format into timestamped Utterances with Indic normalization.
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

            clean_text = IndicNormalizer.normalize_text(text.strip())
            words = len(clean_text.split())
            duration = max(1.5, round(words / 2.5, 1))
            end_time = round(current_time + duration, 1)

            utterances.append(
                Utterance(speaker=speaker, start_time=current_time, end_time=end_time, transcript=clean_text)
            )
            current_time = round(end_time + 0.4, 1)

        return utterances
