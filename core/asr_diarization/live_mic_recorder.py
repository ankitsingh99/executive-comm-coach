"""
Live Microphone Audio Ingestion Engine.
Captures 16kHz 16-bit mono PCM directly from macOS hardware microphone using sounddevice / CoreAudio.
"""

import os
import time
import tempfile
from typing import Optional


class LiveMicRecorder:
    """
    Captures live audio from the physical hardware microphone.
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def record_to_wav(self, duration_seconds: int = 8, output_wav_path: Optional[str] = None) -> str:
        """
        Records live microphone audio for the specified duration (in seconds).
        Returns the path to the recorded 16kHz WAV file.
        """
        if output_wav_path is None:
            temp_dir = tempfile.gettempdir()
            output_wav_path = os.path.join(temp_dir, f"mic_session_{int(time.time())}.wav")

        print(f"  [MICROPHONE ACTIVE] Recording {duration_seconds}s directly from your microphone...")

        # Preferred recording method: sounddevice + scipy
        try:
            import sounddevice as sd
            import scipy.io.wavfile as wav
            import numpy as np

            # Record mono 16kHz audio
            recording = sd.rec(
                int(duration_seconds * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16"
            )
            for remaining in range(duration_seconds, 0, -1):
                print(f"  [SPEAK NOW] {remaining}s remaining...", end="\r", flush=True)
                time.sleep(1)
            sd.wait()
            wav.write(output_wav_path, self.sample_rate, recording)
            print("\n  [CAPTURE COMPLETE] Audio successfully recorded from microphone.")
            return output_wav_path

        except Exception as e:
            # Fallback to ffmpeg avfoundation
            import subprocess
            cmd = [
                "ffmpeg", "-y",
                "-f", "avfoundation",
                "-i", ":0",
                "-t", str(duration_seconds),
                "-ar", str(self.sample_rate),
                "-ac", "1",
                output_wav_path
            ]
            process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for remaining in range(duration_seconds, 0, -1):
                print(f"  [SPEAK NOW] {remaining}s remaining...", end="\r", flush=True)
                time.sleep(1)
            process.wait()
            print("\n  [CAPTURE COMPLETE] Audio successfully recorded via AVFoundation.")
            return output_wav_path
