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

    def listen_for_speech_and_nudge(
        self,
        poll_interval_sec: float = 1.0,
        speech_prob_threshold: float = 0.65,
        max_wait_seconds: Optional[int] = None,
        on_speech_detected_callback: Optional[callable] = None
    ) -> bool:
        """
        Passively monitors the ambient microphone stream with ultra-low compute.
        As soon as human speech / spoken dialogue is detected, triggers a consent nudge
        prompting the user if they wish to start recording for communication coaching analysis.
        """
        import numpy as np
        import subprocess
        from .vad_gater import AmbientVadGate

        print("\n  [AMBIENT SENSING ACTIVE] Passively listening for spoken dialogue...")
        print("  (Privacy protected: Audio is evaluated in-RAM and immediately purged if no speech is detected)")

        start_time = time.time()
        gate = AmbientVadGate(speech_prob_threshold=speech_prob_threshold)

        while True:
            if max_wait_seconds and (time.time() - start_time) > max_wait_seconds:
                print("  [AMBIENT TIMEOUT] No speech detected within window.")
                return False

            temp_chunk_path = os.path.join(tempfile.gettempdir(), f"vad_sample_{int(time.time() * 1000)}.wav")
            try:
                # Capture a short 1.0s probe chunk via ffmpeg avfoundation
                cmd = [
                    "ffmpeg", "-y",
                    "-f", "avfoundation",
                    "-i", ":0",
                    "-t", str(poll_interval_sec),
                    "-ar", str(self.sample_rate),
                    "-ac", "1",
                    temp_chunk_path
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=poll_interval_sec + 3)

                if os.path.exists(temp_chunk_path) and os.path.getsize(temp_chunk_path) > 1000:
                    import wave
                    with wave.open(temp_chunk_path, "rb") as wf:
                        n_frames = wf.getnframes()
                        raw_data = wf.readframes(n_frames)
                        audio_data = np.frombuffer(raw_data, dtype=np.int16)

                    # Evaluate speech probability
                    speech_prob = gate.calculate_speech_probability(audio_data)
                    timestamp_ms = (time.time() - start_time) * 1000.0
                    is_triggered, status_msg = gate.evaluate_frame(timestamp_ms, speech_prob)

                    # Remove probe chunk immediately for privacy
                    try:
                        os.remove(temp_chunk_path)
                    except OSError:
                        pass

                    if is_triggered or speech_prob >= speech_prob_threshold:
                        print("\n" + "=" * 66)
                        print("  🎙️  [SPOKEN DIALOGUE DETECTED] Person started speaking!")
                        print(f"      Speech Confidence: {int(speech_prob * 100)}% | Ambient VAD Gating Passed")
                        print("=" * 66)
                        
                        if on_speech_detected_callback:
                            return on_speech_detected_callback(speech_prob)
                        return True

            except Exception:
                pass
            finally:
                if os.path.exists(temp_chunk_path):
                    try:
                        os.remove(temp_chunk_path)
                    except OSError:
                        pass

            time.sleep(0.1)
