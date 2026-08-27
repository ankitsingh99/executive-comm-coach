"""
Live Microphone Audio Ingestion Engine.
Captures 16kHz 16-bit mono PCM from macOS default microphone via avfoundation / ffmpeg.
"""

import os
import subprocess
import tempfile
import time
from typing import Optional


class LiveMicRecorder:
    """
    Captures live audio from the physical hardware microphone.
    """

    def __init__(self, sample_rate: int = 16000, ffmpeg_path: str = "/opt/homebrew/bin/ffmpeg"):
        self.sample_rate = sample_rate
        self.ffmpeg_path = ffmpeg_path if os.path.exists(ffmpeg_path) else "ffmpeg"

    def record_to_wav(self, duration_seconds: int = 10, output_wav_path: Optional[str] = None) -> str:
        """
        Records live microphone audio for the specified duration (in seconds).
        Returns the path to the recorded 16kHz WAV file.
        """
        if output_wav_path is None:
            temp_dir = tempfile.gettempdir()
            output_wav_path = os.path.join(temp_dir, f"mic_session_{int(time.time())}.wav")

        print(f"  [RECORDING] Microphone ACTIVE ({duration_seconds}s). Speak now into your microphone...")

        # Record from default macOS audio input :0
        cmd = [
            self.ffmpeg_path,
            "-y",
            "-f", "avfoundation",
            "-i", ":0",
            "-t", str(duration_seconds),
            "-ar", str(self.sample_rate),
            "-ac", "1",
            output_wav_path
        ]

        process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Display progress countdown
        for remaining in range(duration_seconds, 0, -1):
            print(f"  [RECORDING] {remaining}s remaining... (Speaking)", end="\r", flush=True)
            time.sleep(1)

        process.wait()
        print("\n  [COMPLETED] Audio capture finished.")
        return output_wav_path
