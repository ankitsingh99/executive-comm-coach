"""
Acoustic Non-Phonetic Filler & Vocal Hesitation Detector.
Analyzes raw audio waveforms to detect non-verbal and non-phonetic hesitation sounds
(e.g., 'umm', 'aah', 'aaaaa', 'uhh', 'hmm', 'err') using pitch stationarity,
spectral flux, zero-crossing rate, and formant/nasal resonance band energy ratios.
"""

import os
import wave
import numpy as np
from typing import List, Optional, Dict, Any

try:
    from ..engine.schema import Utterance, AcousticFillerEvent
except (ImportError, ValueError):
    from engine.schema import Utterance, AcousticFillerEvent


class AcousticFillerDetector:
    """
    On-device DSP engine to detect non-phonetic acoustic filler sounds directly from raw audio.
    Identifies prolonged voiced hesitations and nasal murmurs before or alongside STT.
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.frame_length_ms = 30.0  # 30ms window (480 samples @ 16kHz)
        self.hop_length_ms = 10.0  # 10ms hop (160 samples @ 16kHz)

    def detect_fillers_from_wav(
        self, wav_path: str, speaker_segments: Optional[List[Utterance]] = None
    ) -> List[AcousticFillerEvent]:
        """
        Parses a WAV file and returns all detected acoustic filler events with timestamps.
        """
        if not os.path.exists(wav_path):
            return []

        try:
            with wave.open(wav_path, "rb") as wf:
                num_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                sr = wf.getframerate()
                n_frames = wf.getnframes()
                raw_bytes = wf.readframes(n_frames)

            if sample_width == 2:
                audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            elif sample_width == 4:
                audio = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                audio = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0

            if num_channels > 1:
                audio = audio.reshape(-1, num_channels).mean(axis=1)

            # Resample to self.sample_rate if needed
            if sr != self.sample_rate and len(audio) > 0:
                indices = np.round(np.arange(0, len(audio), sr / self.sample_rate)).astype(int)
                indices = indices[indices < len(audio)]
                audio = audio[indices]

            return self.detect_fillers_from_signal(audio, self.sample_rate, speaker_segments=speaker_segments)

        except Exception:
            return []

    def detect_fillers_from_signal(
        self, signal: np.ndarray, sample_rate: int, speaker_segments: Optional[List[Utterance]] = None
    ) -> List[AcousticFillerEvent]:
        """
        Extracts acoustic hesitation events from float32 mono audio.
        """
        if len(signal) == 0:
            return []

        frame_len = int(self.frame_length_ms * sample_rate / 1000)
        hop_len = int(self.hop_length_ms * sample_rate / 1000)

        # Baseline noise floor estimation (bottom 15% quantile)
        sub_samples = signal[::4]
        if len(sub_samples) > 0:
            noise_floor_rms = float(np.percentile(np.abs(sub_samples), 15))
        else:
            noise_floor_rms = 0.002
        noise_floor_rms = max(0.001, noise_floor_rms)

        min_lag = int(sample_rate / 450)  # max pitch ~450 Hz
        max_lag = int(sample_rate / 75)  # min pitch ~75 Hz

        frames = []
        timestamps = []
        for start in range(0, len(signal) - frame_len + 1, hop_len):
            frames.append(signal[start : start + frame_len])
            timestamps.append(start / float(sample_rate))

        if not frames:
            return []

        # 1. Extract frame-level acoustic properties
        frame_data = []
        prev_mag = None

        for idx, frame in enumerate(frames):
            rms = float(np.sqrt(np.mean(frame**2)))
            zcr = float(np.mean(np.abs(np.diff(np.sign(frame))))) / 2.0

            # Windowed FFT
            windowed = frame * np.hanning(len(frame))
            fft_complex = np.fft.rfft(windowed)
            fft_mag = np.abs(fft_complex)
            freqs = np.fft.rfftfreq(len(frame), 1.0 / sample_rate)

            # Spectral flux (spectral stationarity measure)
            if prev_mag is not None and np.sum(fft_mag) > 1e-5 and np.sum(prev_mag) > 1e-5:
                norm_cur = fft_mag / (np.sum(fft_mag) + 1e-6)
                norm_prev = prev_mag / (np.sum(prev_mag) + 1e-6)
                flux = float(np.sum(np.abs(norm_cur - norm_prev)))
            else:
                flux = 0.5
            prev_mag = fft_mag

            mag_sum = float(np.sum(fft_mag))
            centroid = float(np.sum(freqs * fft_mag) / mag_sum) if mag_sum > 1e-6 else 0.0

            # Energy bands
            low_band = float(np.sum(fft_mag[(freqs >= 100) & (freqs <= 450)] ** 2))
            mid_band = float(np.sum(fft_mag[(freqs > 450) & (freqs <= 1800)] ** 2))
            high_band = float(np.sum(fft_mag[freqs > 2500] ** 2))
            total_band = low_band + mid_band + high_band + 1e-6

            low_ratio = low_band / total_band
            mid_ratio = mid_band / total_band
            high_ratio = high_band / total_band

            # Pitch via Normalized Autocorrelation
            is_voiced = False
            pitch_f0 = 0.0
            if rms > max(0.008, noise_floor_rms * 1.8) and zcr < 0.18:
                corr = np.correlate(frame, frame, mode="full")
                corr = corr[len(corr) // 2 :]
                if len(corr) > max_lag and corr[0] > 1e-6:
                    search_slice = corr[min_lag:max_lag]
                    if len(search_slice) > 0:
                        peak_val = np.max(search_slice)
                        peak_idx = min_lag + np.argmax(search_slice)
                        if peak_val > 0.35 * corr[0]:
                            is_voiced = True
                            pitch_f0 = float(sample_rate / peak_idx)

            # Hesitation candidate condition:
            # Voiced, low ZCR (< 0.14), low spectral flux (< 0.28 = stationary sound), low high-frequency energy (< 0.18)
            is_hesitation_candidate = (
                is_voiced
                and (zcr <= 0.14)
                and (flux <= 0.28)
                and (high_ratio <= 0.18)
                and (rms >= max(0.009, noise_floor_rms * 2.0))
            )

            frame_data.append(
                {
                    "time": timestamps[idx],
                    "rms": rms,
                    "zcr": zcr,
                    "flux": flux,
                    "centroid": centroid,
                    "low_ratio": low_ratio,
                    "mid_ratio": mid_ratio,
                    "is_voiced": is_voiced,
                    "pitch_f0": pitch_f0,
                    "is_candidate": is_hesitation_candidate,
                }
            )

        # 2. Group contiguous candidate frames into candidate hesitation intervals
        events: List[AcousticFillerEvent] = []
        in_segment = False
        seg_frames = []

        for f in frame_data:
            if f["is_candidate"]:
                in_segment = True
                seg_frames.append(f)
            else:
                if in_segment and seg_frames:
                    event = self._evaluate_segment(seg_frames, speaker_segments)
                    if event is not None:
                        events.append(event)
                    seg_frames = []
                in_segment = False

        if in_segment and seg_frames:
            event = self._evaluate_segment(seg_frames, speaker_segments)
            if event is not None:
                events.append(event)

        return events

    def _evaluate_segment(
        self, frames: List[Dict[str, Any]], speaker_segments: Optional[List[Utterance]] = None
    ) -> Optional[AcousticFillerEvent]:
        """
        Evaluates a contiguous cluster of stationary voiced frames to determine if it is a true filler.
        """
        if not frames:
            return None

        start_time = round(frames[0]["time"], 2)
        end_time = round(frames[-1]["time"] + (self.frame_length_ms / 1000.0), 2)
        duration_sec = round(end_time - start_time, 2)

        # Non-phonetic hesitation sounds typically last between 0.22s and 1.8s
        if duration_sec < 0.22 or duration_sec > 2.0:
            return None

        pitches = [f["pitch_f0"] for f in frames if f["pitch_f0"] > 0]
        if not pitches:
            return None

        # Flat pitch check: hesitations have remarkably steady F0 (standard deviation < 22 Hz)
        pitch_std = float(np.std(pitches))
        if pitch_std > 22.0 and duration_sec < 0.5:
            return None

        mean_centroid = float(np.mean([f["centroid"] for f in frames]))
        mean_low_ratio = float(np.mean([f["low_ratio"] for f in frames]))
        mean_mid_ratio = float(np.mean([f["mid_ratio"] for f in frames]))
        mean_flux = float(np.mean([f["flux"] for f in frames]))

        # Strict stationarity check
        if mean_flux > 0.25:
            return None

        # True nasal murmurs ('umm', 'hmm') have closed oral cavity, thus very low mid-band energy (< 0.22)
        if mean_mid_ratio < 0.22 and (mean_low_ratio >= 0.65 or mean_centroid < 420.0):
            token = "umm"
            acoustic_type = "nasal_murmur"
            confidence = min(0.96, 0.82 + (mean_low_ratio * 0.2))
        elif duration_sec >= 0.45:
            token = "aaaaa"
            acoustic_type = "vowel_elongation"
            confidence = 0.93
        elif mean_centroid <= 1200.0 or mean_mid_ratio >= 0.35:
            token = "aah"
            acoustic_type = "vocal_hesitation"
            confidence = 0.89
        elif 1200.0 < mean_centroid <= 1900.0:
            token = "uhh"
            acoustic_type = "open_pause"
            confidence = 0.88
        else:
            token = "err"
            acoustic_type = "vocal_hesitation"
            confidence = 0.85

        # Determine speaker mapping
        assigned_speaker = "USER"
        if speaker_segments:
            for utt in speaker_segments:
                if utt.start_time <= start_time <= utt.end_time or utt.start_time <= end_time <= utt.end_time:
                    assigned_speaker = utt.speaker
                    break

        return AcousticFillerEvent(
            token=token,
            start_time=start_time,
            end_time=end_time,
            duration_sec=duration_sec,
            confidence=round(confidence, 2),
            speaker=assigned_speaker,
            acoustic_type=acoustic_type,
        )

    @classmethod
    def inject_fillers_into_utterances(
        cls, utterances: List[Utterance], acoustic_fillers: List[AcousticFillerEvent]
    ) -> List[Utterance]:
        """
        Merges detected acoustic fillers into the transcribed utterances at their respective timestamps
        if the STT engine omitted or cleaned them up.
        """
        if not acoustic_fillers or not utterances:
            return utterances

        updated_utterances = []
        for utt in utterances:
            # Find acoustic fillers that occurred during this utterance
            utt_fillers = [
                f
                for f in acoustic_fillers
                if (utt.start_time - 0.25) <= f.start_time <= (utt.end_time + 0.25)
                and (f.speaker == utt.speaker or f.speaker in ["USER", "SELF"])
            ]

            transcript = utt.transcript
            for fil in utt_fillers:
                # Check if filler is already explicitly transcribed in text
                fil_token = fil.token.lower()
                if not (fil_token in transcript.lower() or f"{fil_token[:2]}" in transcript.lower()):
                    # Inject filler naturally based on relative position
                    rel_pos = (fil.start_time - utt.start_time) / max(0.5, utt.end_time - utt.start_time)
                    if rel_pos <= 0.25:
                        transcript = f"{fil.token} {transcript}"
                    elif rel_pos >= 0.75:
                        transcript = f"{transcript} {fil.token}"
                    else:
                        words = transcript.split()
                        mid = len(words) // 2
                        words.insert(mid, fil.token)
                        transcript = " ".join(words)

            updated_utterances.append(
                Utterance(
                    speaker=utt.speaker,
                    start_time=utt.start_time,
                    end_time=utt.end_time,
                    transcript=transcript.strip(),
                )
            )

        return updated_utterances
