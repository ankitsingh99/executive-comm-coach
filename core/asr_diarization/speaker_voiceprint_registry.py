"""
Persistent Acoustic Voiceprint Memory & Speaker Recognition Registry.
Extracts on-device acoustic voice embeddings (pitch contours, spectral timbre, formant distributions)
to enroll, identify, and recall known speakers across conversations without manual re-tagging.
"""

import os
import json
import wave
import numpy as np
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Any

try:
    from ..config import DATA_DIR
    from ..engine.persona_ontology import PowerAxis
except (ImportError, ValueError):
    from config import DATA_DIR
    from engine.persona_ontology import PowerAxis


@dataclass
class SpeakerVoiceprint:
    """Acoustic biometric voiceprint profile for a known person."""
    speaker_name: str
    role: str = "Colleague"
    power_axis: str = "LATERAL"
    embedding_vector: List[float] = field(default_factory=list)
    mean_pitch_hz: float = 150.0
    pitch_range_hz: float = 35.0
    spectral_centroid_hz: float = 1800.0
    enrolled_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    sample_count: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpeakerVoiceprint":
        return cls(
            speaker_name=data.get("speaker_name", "Unknown"),
            role=data.get("role", "Colleague"),
            power_axis=data.get("power_axis", "LATERAL"),
            embedding_vector=data.get("embedding_vector", []),
            mean_pitch_hz=float(data.get("mean_pitch_hz", 150.0)),
            pitch_range_hz=float(data.get("pitch_range_hz", 35.0)),
            spectral_centroid_hz=float(data.get("spectral_centroid_hz", 1800.0)),
            enrolled_at_utc=data.get("enrolled_at_utc", datetime.now(timezone.utc).isoformat()),
            sample_count=int(data.get("sample_count", 1))
        )


class SpeakerVoiceprintRegistry:
    """
    On-device persistent voiceprint memory database.
    Stores and matches voice acoustic embeddings using cosine similarity.
    """

    def __init__(self, storage_dir: Optional[str] = None):
        self.storage_dir = storage_dir or os.path.join(DATA_DIR, "speaker_vault")
        os.makedirs(self.storage_dir, exist_ok=True)
        self.registry_file = os.path.join(self.storage_dir, "voiceprints.json")
        self.voiceprints: Dict[str, SpeakerVoiceprint] = {}
        self.load_from_disk()

    def load_from_disk(self):
        """Loads enrolled voiceprints from local JSON vault."""
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.voiceprints = {
                        k: SpeakerVoiceprint.from_dict(v) for k, v in data.items()
                    }
            except Exception:
                self.voiceprints = {}
        else:
            self.voiceprints = {}

    def save_to_disk(self):
        """Persists enrolled voiceprints to local vault."""
        try:
            data = {k: v.to_dict() for k, v in self.voiceprints.items()}
            with open(self.registry_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def extract_voiceprint_features(
        self,
        audio_signal: np.ndarray,
        sample_rate: int = 16000
    ) -> Optional[Tuple[List[float], float, float, float]]:
        """
        Extracts 32-dimensional normalized acoustic voiceprint vector
        along with fundamental pitch F0, pitch variance, and spectral centroid.
        """
        if len(audio_signal) < int(sample_rate * 0.3):  # Less than 300ms
            return None

        frame_len = int(0.040 * sample_rate)  # 40ms frame
        hop_len = int(0.020 * sample_rate)    # 20ms hop

        frames = []
        for start in range(0, len(audio_signal) - frame_len + 1, hop_len):
            frame = audio_signal[start : start + frame_len]
            rms = np.sqrt(np.mean(frame**2))
            if rms > 0.012:  # Voice activity threshold
                frames.append(frame)

        if not frames or len(frames) < 5:
            return None

        # 1. Pitch Estimation via Autocorrelation across voiced frames
        pitches = []
        min_lag = int(sample_rate / 450)  # Max pitch: 450 Hz
        max_lag = int(sample_rate / 75)   # Min pitch: 75 Hz

        for f in frames:
            corr = np.correlate(f, f, mode="full")[len(f) - 1 :]
            if len(corr) > max_lag:
                slice_corr = corr[min_lag:max_lag]
                if len(slice_corr) > 0 and np.max(slice_corr) > 0.35 * corr[0]:
                    peak_idx = min_lag + np.argmax(slice_corr)
                    f0 = sample_rate / peak_idx
                    pitches.append(f0)

        mean_pitch = float(np.mean(pitches)) if pitches else 150.0
        pitch_range = float(np.ptp(pitches)) if len(pitches) > 1 else 30.0

        # 2. Multi-Band Spectral Energy & Timbre Profile (24 frequency bands)
        band_energies = []
        centroids = []
        fft_len = 512

        # Logarithmic filter bank bounds (80 Hz to 7500 Hz)
        freq_bins = np.logspace(np.log10(80), np.log10(sample_rate / 2.0), num=25)

        for f in frames:
            # Windowing & FFT
            windowed = f * np.hanning(len(f))
            fft_mag = np.abs(np.fft.rfft(windowed, n=fft_len))
            freqs = np.fft.rfftfreq(fft_len, 1.0 / sample_rate)

            # Spectral Centroid
            mag_sum = np.sum(fft_mag)
            if mag_sum > 1e-6:
                c = np.sum(freqs * fft_mag) / mag_sum
                centroids.append(c)

            # Filter bank binning
            frame_bands = []
            for b in range(len(freq_bins) - 1):
                idx = np.where((freqs >= freq_bins[b]) & (freqs < freq_bins[b + 1]))[0]
                if len(idx) > 0:
                    band_energy = np.mean(fft_mag[idx])
                else:
                    band_energy = 0.0
                frame_bands.append(band_energy)
            band_energies.append(frame_bands)

        mean_centroid = float(np.mean(centroids)) if centroids else 1800.0
        avg_bands = np.mean(band_energies, axis=0)
        std_bands = np.std(band_energies, axis=0) if len(band_energies) > 1 else np.zeros_like(avg_bands)

        # 3. Assemble 32-D Feature Vector
        # [24 normalized band energies, 4 band dynamics, normalized pitch, normalized centroid, pitch range, energy skew]
        norm_bands = avg_bands / (np.linalg.norm(avg_bands) + 1e-6)
        dynamics = [
            float(np.mean(std_bands[:6])),
            float(np.mean(std_bands[6:12])),
            float(np.mean(std_bands[12:18])),
            float(np.mean(std_bands[18:]))
        ]
        features = list(norm_bands) + dynamics + [
            float(mean_pitch / 400.0),
            float(pitch_range / 200.0),
            float(mean_centroid / 4000.0),
            float(np.std(pitches) / 100.0 if len(pitches) > 1 else 0.1)
        ]

        # L2-normalize final embedding vector
        vec = np.array(features, dtype=np.float32)
        norm_vec = (vec / (np.linalg.norm(vec) + 1e-6)).tolist()

        return norm_vec, mean_pitch, pitch_range, mean_centroid

    def _load_audio_file(self, wav_path: str) -> Optional[Tuple[np.ndarray, int]]:
        """Reads WAV audio file into float32 numpy array."""
        if not os.path.exists(wav_path):
            return None
        try:
            with wave.open(wav_path, "rb") as wf:
                num_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                sr = wf.getframerate()
                n_frames = wf.getnframes()
                raw_bytes = wf.readframes(n_frames)

            if sample_width == 2:
                signal = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            elif sample_width == 4:
                signal = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                signal = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0

            if num_channels > 1:
                signal = signal.reshape(-1, num_channels).mean(axis=1)

            return signal, sr
        except Exception:
            return None

    def enroll_speaker(
        self,
        name: str,
        role: str = "Colleague",
        power_axis: str = "LATERAL",
        audio_signal_or_wav_path: Any = None,
        sample_rate: int = 16000
    ) -> Optional[SpeakerVoiceprint]:
        """
        Enrolls or updates a speaker's voiceprint in the persistent registry.
        """
        if isinstance(audio_signal_or_wav_path, str):
            res = self._load_audio_file(audio_signal_or_wav_path)
            if res is None:
                return None
            signal, sr = res
        elif isinstance(audio_signal_or_wav_path, np.ndarray):
            signal = audio_signal_or_wav_path
            sr = sample_rate
        else:
            return None

        extracted = self.extract_voiceprint_features(signal, sr)
        if extracted is None:
            return None

        embedding, mean_pitch, pitch_range, centroid = extracted
        key = name.strip()

        if key in self.voiceprints:
            # Update existing voiceprint with running average
            existing = self.voiceprints[key]
            n = existing.sample_count
            old_vec = np.array(existing.embedding_vector, dtype=np.float32)
            new_vec = np.array(embedding, dtype=np.float32)
            merged_vec = (old_vec * n + new_vec) / (n + 1)
            merged_vec = (merged_vec / (np.linalg.norm(merged_vec) + 1e-6)).tolist()

            voiceprint = SpeakerVoiceprint(
                speaker_name=name.strip(),
                role=role or existing.role,
                power_axis=power_axis or existing.power_axis,
                embedding_vector=merged_vec,
                mean_pitch_hz=round((existing.mean_pitch_hz * n + mean_pitch) / (n + 1), 1),
                pitch_range_hz=round((existing.pitch_range_hz * n + pitch_range) / (n + 1), 1),
                spectral_centroid_hz=round((existing.spectral_centroid_hz * n + centroid) / (n + 1), 1),
                sample_count=n + 1
            )
        else:
            voiceprint = SpeakerVoiceprint(
                speaker_name=name.strip(),
                role=role,
                power_axis=power_axis,
                embedding_vector=embedding,
                mean_pitch_hz=round(mean_pitch, 1),
                pitch_range_hz=round(pitch_range, 1),
                spectral_centroid_hz=round(centroid, 1),
                sample_count=1
            )

        self.voiceprints[key] = voiceprint
        self.save_to_disk()
        return voiceprint

    def identify_speaker(
        self,
        audio_signal_or_wav_path: Any,
        sample_rate: int = 16000,
        threshold: float = 0.78
    ) -> Optional[Tuple[SpeakerVoiceprint, float]]:
        """
        Matches incoming audio signal against enrolled voiceprints.
        Returns (Matched SpeakerVoiceprint, confidence_score) if match exceeds threshold.
        """
        if not self.voiceprints:
            return None

        if isinstance(audio_signal_or_wav_path, str):
            res = self._load_audio_file(audio_signal_or_wav_path)
            if res is None:
                return None
            signal, sr = res
        elif isinstance(audio_signal_or_wav_path, np.ndarray):
            signal = audio_signal_or_wav_path
            sr = sample_rate
        else:
            return None

        extracted = self.extract_voiceprint_features(signal, sr)
        if extracted is None:
            return None

        query_vec, query_pitch, _, query_centroid = extracted
        q_arr = np.array(query_vec, dtype=np.float32)

        best_speaker: Optional[SpeakerVoiceprint] = None
        best_score: float = -1.0

        for spk in self.voiceprints.values():
            ref_arr = np.array(spk.embedding_vector, dtype=np.float32)
            cosine_sim = float(np.dot(q_arr, ref_arr) / (np.linalg.norm(q_arr) * np.linalg.norm(ref_arr) + 1e-6))

            # Pitch compatibility penalty: if fundamental pitch differs by > 75 Hz, apply soft penalty
            pitch_diff = abs(query_pitch - spk.mean_pitch_hz)
            pitch_factor = 1.0 if pitch_diff < 45.0 else max(0.65, 1.0 - (pitch_diff - 45.0) / 150.0)

            final_score = cosine_sim * pitch_factor
            if final_score > best_score:
                best_score = final_score
                best_speaker = spk

        if best_speaker is not None and best_score >= threshold:
            return best_speaker, round(best_score, 3)

        return None

    def list_enrolled_speakers(self) -> List[Dict[str, Any]]:
        """Returns summary metadata of all enrolled voiceprints."""
        self.load_from_disk()
        return [
            {
                "name": v.speaker_name,
                "role": v.role,
                "power_axis": v.power_axis,
                "mean_pitch_hz": v.mean_pitch_hz,
                "sample_count": v.sample_count,
                "enrolled_at_utc": v.enrolled_at_utc
            }
            for v in self.voiceprints.values()
        ]

    def delete_voiceprint(self, speaker_name: str) -> bool:
        """Deletes an enrolled voiceprint (DPDP Section 12 Right to Erasure)."""
        self.load_from_disk()
        target = speaker_name.strip()
        matched = None
        for k in self.voiceprints.keys():
            if k.lower() == target.lower():
                matched = k
                break

        if matched:
            del self.voiceprints[matched]
            self.save_to_disk()
            return True
        return False
