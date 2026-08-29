"""
NVIDIA Parakeet State-of-the-Art On-Device Speech Recognition Engine.
Executes NVIDIA's Parakeet CTC/TDT Conformer model for fast, high-accuracy acoustic transcription.
"""

import os
from typing import List, Optional

try:
    import torch
    import librosa
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    librosa = None
    TORCH_AVAILABLE = False

try:
    from ..engine.schema import Utterance
except (ImportError, ValueError):
    from engine.schema import Utterance


class NvidiaParakeetEngine:
    """
    NVIDIA Parakeet Speech-to-Text inference engine.
    Uses 'nvidia/parakeet-ctc-0.6b' with local Metal / CPU acceleration.
    """

    def __init__(self, model_id: str = "nvidia/parakeet-ctc-0.6b"):
        self.model_id = model_id
        if TORCH_AVAILABLE and torch is not None:
            self.device = "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
        else:
            self.device = "cpu"
        self._processor = None
        self._model = None

    def _load_model(self):
        if self._model is None or self._processor is None:
            from transformers import AutoProcessor, AutoModelForCTC
            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._model = AutoModelForCTC.from_pretrained(self.model_id).to(self.device)
            self._model.eval()

    def transcribe_audio_file(self, audio_wav_path: str, speaker_id: str = "USER") -> List[Utterance]:
        """
        Transcribes a 16kHz WAV audio file using NVIDIA Parakeet.
        """
        if not os.path.exists(audio_wav_path):
            return []

        try:
            self._load_model()
            # Load audio resampled to 16kHz mono
            speech, sr = librosa.load(audio_wav_path, sr=16000, mono=True)
            if len(speech) < 1600:  # Less than 0.1s
                return []

            inputs = self._processor(speech, sampling_rate=16000, return_tensors="pt")
            input_values = inputs.input_features.to(self.device)

            with torch.no_grad():
                logits = self._model(input_values).logits

            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self._processor.batch_decode(predicted_ids)[0].strip()

            if not transcription:
                return []

            duration = round(len(speech) / 16000.0, 2)
            return [
                Utterance(
                    speaker=speaker_id,
                    start_time=0.0,
                    end_time=duration,
                    transcript=transcription
                )
            ]
        except Exception as e:
            return []
