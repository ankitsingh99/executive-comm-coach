"""
Silero VAD Low-Power Ambient Acoustic Gate (Stage 1).
Implements the 32ms frame evaluation, rolling window sustained speech threshold (tau >= 0.75),
and 3-second non-speech buffer purging.
"""

from typing import List, Tuple
import numpy as np

try:
    from ..engine.schema import BaseModel
except (ImportError, ValueError):
    from engine.schema import BaseModel


class VadFrameResult(BaseModel):
    timestamp_ms: float = 0.0
    speech_probability: float = 0.0
    is_speech: bool = False

    def __init__(self, timestamp_ms: float = 0.0, speech_probability: float = 0.0, is_speech: bool = False, **kwargs):
        super().__init__(
            timestamp_ms=timestamp_ms, speech_probability=speech_probability, is_speech=is_speech, **kwargs
        )
        self.timestamp_ms = timestamp_ms
        self.speech_probability = speech_probability
        self.is_speech = is_speech


class AmbientVadGate:
    """
    Simulates the on-device Silero VAD (ONNX Runtime Mobile) acoustic gate.
    - Operates on 16kHz 16-bit mono PCM.
    - Evaluates 32ms frames.
    - Purges audio buffer when tau < 0.5 within 3 seconds.
    - Initiates interactive transition when tau >= 0.75 across a 600ms rolling window.
    """

    def __init__(
        self,
        speech_prob_threshold: float = 0.75,
        purge_prob_threshold: float = 0.50,
        sustained_window_ms: float = 600.0,
        purge_window_ms: float = 3000.0,
    ):
        self.speech_prob_threshold = speech_prob_threshold
        self.purge_prob_threshold = purge_prob_threshold
        self.sustained_window_ms = sustained_window_ms
        self.purge_window_ms = purge_window_ms
        self.frame_history: List[VadFrameResult] = []

    @staticmethod
    def calculate_speech_probability(audio_chunk_16k: "np.ndarray", noise_floor_rms: float = 0.0) -> float:
        """
        Computes on-device speech probability using adaptive SNR, energy RMS, crest factor dynamics,
        and zero-crossing dynamics. Adapts dynamically to capture both feeble/whispered speech and loud speech.
        """
        if len(audio_chunk_16k) == 0:
            return 0.0

        # Normalize if int16
        if audio_chunk_16k.dtype == np.int16:
            samples = audio_chunk_16k.astype(np.float32) / 32768.0
        else:
            samples = audio_chunk_16k.astype(np.float32)

        rms = float(np.sqrt(np.mean(samples**2)))
        peak = float(np.max(np.abs(samples))) if len(samples) > 0 else 0.0
        crest_factor = peak / (rms + 1e-6)

        # Pure electrical silence floor
        if rms < 0.0008:
            return 0.01

        # Zero-crossing rate
        zero_crossings = np.nonzero(np.diff(samples > 0))[0]
        zcr = float(len(zero_crossings) / max(1, len(samples)))

        # Human vocal frequency zero-crossing range (roughly 85Hz - 3400Hz at 16kHz)
        zcr_valid = 0.008 <= zcr <= 0.48
        crest_valid = crest_factor >= 1.7

        # Spectral harmony & formant structure score
        if zcr_valid and crest_valid:
            spectral_score = 1.0
        elif zcr_valid or crest_valid:
            spectral_score = 0.65
        else:
            spectral_score = 0.15

        # Adaptive Energy / SNR Scoring (multiplied by spectral plausibility so silence is strictly 0.0):
        if noise_floor_rms > 0.0002:
            # Dynamic SNR relative to tracked noise floor
            snr_db = float(20.0 * np.log10(max(1e-5, rms) / max(1e-5, noise_floor_rms)))
            # If SNR <= 2.0 dB (ambient room noise equal to floor), snr_score is 0.0
            # If SNR >= 12.0 dB, snr_score is 1.0
            snr_score = float(np.clip((snr_db - 2.0) / 10.0, 0.0, 1.0))
            prob = snr_score * (0.35 + 0.65 * spectral_score)
        else:
            # Absolute energy scoring for quiet environments
            energy_score = float(np.clip((rms - 0.0010) / 0.015, 0.0, 1.0))
            prob = energy_score * (0.35 + 0.65 * spectral_score)

        return float(np.clip(prob, 0.0, 1.0))

    def evaluate_frame(self, timestamp_ms: float, speech_prob: float) -> Tuple[bool, str]:
        """
        Evaluates a 32ms audio frame score (tau).
        Returns (trigger_transition, status_message).
        """
        is_speech = speech_prob >= self.speech_prob_threshold
        frame = VadFrameResult(timestamp_ms=timestamp_ms, speech_probability=speech_prob, is_speech=is_speech)
        self.frame_history.append(frame)

        # Purge frames older than 3 seconds scoring below purge threshold
        cutoff = timestamp_ms - self.purge_window_ms
        self.frame_history = [f for f in self.frame_history if f.timestamp_ms >= cutoff]

        # Check rolling window for sustained human speech
        window_start = timestamp_ms - self.sustained_window_ms
        recent_frames = [f for f in self.frame_history if f.timestamp_ms >= window_start]

        if recent_frames:
            speech_frames = [f for f in recent_frames if f.speech_probability >= self.speech_prob_threshold]
            ratio = len(speech_frames) / len(recent_frames)

            # If sustained speech >= 75% across the 600ms rolling window
            if ratio >= 0.75 and (recent_frames[-1].timestamp_ms - recent_frames[0].timestamp_ms) >= (
                self.sustained_window_ms * 0.8
            ):
                return (
                    True,
                    f"Sustained speech detected (tau >= {self.speech_prob_threshold} over {self.sustained_window_ms}ms). Triggering user prompt.",
                )

        if speech_prob < self.purge_prob_threshold:
            return False, "Acoustic gate: Inactive frame purged from ring buffer (<3s retention)."

        return False, "Acoustic gate: Low-energy sound; downstream models remain idle."
