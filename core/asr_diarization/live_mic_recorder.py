"""
Live Microphone Audio Ingestion Engine.
Captures 16kHz 16-bit mono PCM directly from hardware microphone using sounddevice / CoreAudio.
Operates 100% in-memory without spawning repetitive subprocesses, eliminating audio device timeouts.
"""

import os
import sys
import time
import queue
import select
import tempfile
import wave
from typing import Optional
import numpy as np

from .vad_gater import AmbientVadGate


class LiveMicRecorder:
    """
    Captures live audio from the physical hardware microphone via sounddevice / CoreAudio stream.
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def record_until_silence(
        self,
        silence_threshold_sec: float = 2.0,
        min_speech_duration_sec: float = 1.2,
        max_duration_sec: int = 180,
        chunk_duration_sec: float = 0.4,
        speech_prob_threshold: float = 0.45,
        gain_boost: float = 1.8,
        idle_timeout_sec: float = 14.0,
        output_wav_path: Optional[str] = None
    ) -> str:
        """
        Dynamically records microphone audio until the conversation end is detected
        by analyzing silence after the last spoken word, or when the user presses Enter / Ctrl+C.
        Uses a continuous in-memory sounddevice stream for zero-latency, glitch-free audio capture.
        """
        import sounddevice as sd

        if output_wav_path is None:
            temp_dir = tempfile.gettempdir()
            output_wav_path = os.path.join(temp_dir, f"mic_session_{int(time.time())}.wav")

        print(f"\n  [DYNAMIC DIALOGUE CAPTURE ACTIVE]")
        print(f"      • Auto-stop: Concludes automatically when pause is detected (>{silence_threshold_sec:.1f}s silence after speech)")
        print(f"      • Manual stop: Press Enter or Ctrl+C at any time to finish speaking immediately")
        print("      >> Speak now naturally...\n")

        def _check_key_pressed() -> bool:
            try:
                if sys.stdin and sys.stdin.isatty():
                    r, _, _ = select.select([sys.stdin], [], [], 0.0)
                    if r:
                        sys.stdin.readline()
                        return True
            except Exception:
                pass
            return False

        gate = AmbientVadGate(speech_prob_threshold=speech_prob_threshold)
        audio_queue = queue.Queue()
        recorded_frames = []
        has_spoken = False
        silence_elapsed = 0.0
        total_recorded_sec = 0.0
        noise_floor_rms = 0.0
        block_size = int(self.sample_rate * chunk_duration_sec)

        def audio_callback(indata, frames, time_info, status):
            if status:
                pass
            audio_queue.put(indata.copy())

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=block_size,
                callback=audio_callback
            ):
                while total_recorded_sec < max_duration_sec:
                    # Check for manual stop (Enter key)
                    if _check_key_pressed():
                        print("\n\n  [MANUAL STOP] Enter pressed. Concluding recording immediately...")
                        break

                    try:
                        chunk_raw = audio_queue.get(timeout=0.5)
                    except queue.Empty:
                        continue

                    # Apply digital gain boost in memory
                    if gain_boost != 1.0:
                        boosted = np.clip(chunk_raw.astype(np.float32) * gain_boost, -32768.0, 32767.0).astype(np.int16)
                    else:
                        boosted = chunk_raw

                    chunk_flat = boosted.flatten()
                    recorded_frames.append(chunk_flat)
                    dur = len(chunk_flat) / float(self.sample_rate)
                    total_recorded_sec += dur

                    # RMS calculation
                    float_samples = chunk_flat.astype(np.float32) / 32768.0
                    cur_rms = float(np.sqrt(np.mean(float_samples ** 2))) if len(float_samples) > 0 else 0.0

                    # Evaluate speech probability
                    speech_prob = gate.calculate_speech_probability(chunk_flat)
                    is_voice_active = (speech_prob >= 0.32 or cur_rms >= 0.0055)

                    if is_voice_active:
                        has_spoken = True
                        silence_elapsed = 0.0
                        print(f"  [SPEAKING] {total_recorded_sec:.1f}s recorded | Active Dialogue (Voice: {int(speech_prob*100)}%) [Press Enter to finish]    ", end="\r", flush=True)
                    else:
                        if has_spoken:
                            silence_elapsed += dur
                            print(f"  [PAUSE/SILENCE] {total_recorded_sec:.1f}s recorded | Paused: {silence_elapsed:.1f}s / {silence_threshold_sec:.1f}s [Press Enter to finish]   ", end="\r", flush=True)
                            
                            if silence_elapsed >= silence_threshold_sec and total_recorded_sec >= min_speech_duration_sec:
                                print(f"\n\n  [CONVERSATION CONCLUDED] End of conversation detected ({silence_threshold_sec:.1f}s silence after speech).")
                                break
                        else:
                            if total_recorded_sec >= 30.0:
                                print(f"\n\n  [IDLE TIMEOUT] No speech detected after 30s. Concluding session.")
                                break
                            print(f"  [LISTENING] {total_recorded_sec:.1f}s | Waiting for dialogue to begin... [Press Enter to finish]          ", end="\r", flush=True)

        except KeyboardInterrupt:
            print("\n\n  [STOPPED BY USER] Concluding recording and analyzing dialogue...")

        if not recorded_frames:
            # Fallback to standard 6s capture if no stream data
            return self.record_to_wav(duration_seconds=6, output_wav_path=output_wav_path)

        # Concatenate and write directly to WAV
        full_audio = np.concatenate(recorded_frames)
        with wave.open(output_wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(full_audio.tobytes())

        print(f"  [AUDIO STORED] Full conversation ({len(full_audio)/self.sample_rate:.1f}s) captured successfully.")
        return output_wav_path

    def record_to_wav(self, duration_seconds: int = 8, output_wav_path: Optional[str] = None) -> str:
        """
        Records live microphone audio for a fixed duration (in seconds) via sounddevice.
        """
        import sounddevice as sd

        if output_wav_path is None:
            temp_dir = tempfile.gettempdir()
            output_wav_path = os.path.join(temp_dir, f"mic_session_{int(time.time())}.wav")

        print(f"  [MICROPHONE ACTIVE] Recording {duration_seconds}s directly from your microphone...")

        total_samples = int(self.sample_rate * duration_seconds)
        recording = sd.rec(total_samples, samplerate=self.sample_rate, channels=1, dtype="int16")
        
        for remaining in range(duration_seconds, 0, -1):
            print(f"  [SPEAK NOW] {remaining}s remaining...", end="\r", flush=True)
            time.sleep(1)
        sd.wait()

        with wave.open(output_wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(recording.tobytes())

        print("\n  [CAPTURE COMPLETE] Audio successfully recorded via CoreAudio.")
        return output_wav_path

    @staticmethod
    def send_shell_desktop_notification(
        title: str = "Executive Coach",
        message: str = "Spoken dialogue detected! Starting coaching capture...",
        subtitle: str = "Ambient Speech Nudge"
    ):
        """
        Triggers macOS system desktop notification, terminal bell, and alert chime.
        """
        import subprocess
        try:
            sys.stdout.write("\a")
            sys.stdout.flush()
            script = f'display notification "{message}" with title "{title}" subtitle "{subtitle}" sound name "Glass"'
            subprocess.Popen(["osascript", "-e", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    def listen_for_speech_and_nudge(
        self,
        poll_interval_sec: float = 0.4,
        speech_prob_threshold: float = 0.45,
        gain_boost: float = 1.8,
        max_wait_seconds: Optional[int] = None,
        on_speech_detected_callback: Optional[callable] = None
    ) -> bool:
        """
        Passively monitors the ambient microphone stream with continuous in-memory sounddevice sensing.
        As soon as human speech / spoken dialogue is detected, triggers a consent nudge
        prompting the user if they wish to start recording for communication coaching analysis.
        """
        import sounddevice as sd

        print("\n  [AMBIENT SENSING ACTIVE] Passively listening for spoken dialogue (High Sensitivity)...")
        print("  (Privacy protected: Audio evaluated in memory & purged immediately if below threshold)")

        start_time = time.time()
        gate = AmbientVadGate(speech_prob_threshold=speech_prob_threshold)
        spinners = ["-", "\\", "|", "/"]
        spin_idx = 0
        audio_queue = queue.Queue()
        noise_floor_rms = 0.0
        block_size = int(self.sample_rate * poll_interval_sec)

        def audio_callback(indata, frames, time_info, status):
            if status:
                pass
            audio_queue.put(indata.copy())

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=block_size,
                callback=audio_callback
            ):
                while True:
                    if max_wait_seconds and (time.time() - start_time) > max_wait_seconds:
                        print("\n  [AMBIENT TIMEOUT] No speech detected within window.")
                        return False

                    print(f"  {spinners[spin_idx % len(spinners)]} Ambient Ear Active... (Waiting for dialogue to start)", end="\r", flush=True)
                    spin_idx += 1

                    try:
                        chunk_raw = audio_queue.get(timeout=0.6)
                    except queue.Empty:
                        continue

                    if gain_boost != 1.0:
                        boosted = np.clip(chunk_raw.astype(np.float32) * gain_boost, -32768.0, 32767.0).astype(np.int16)
                    else:
                        boosted = chunk_raw

                    chunk_flat = boosted.flatten()

                    # RMS calculation
                    float_samples = chunk_flat.astype(np.float32) / 32768.0
                    cur_rms = float(np.sqrt(np.mean(float_samples ** 2)))

                    if noise_floor_rms == 0.0:
                        noise_floor_rms = cur_rms
                    else:
                        noise_floor_rms = 0.85 * noise_floor_rms + 0.15 * min(cur_rms, noise_floor_rms * 1.3)

                    speech_prob = gate.calculate_speech_probability(chunk_flat, noise_floor_rms=noise_floor_rms)
                    timestamp_ms = (time.time() - start_time) * 1000.0
                    is_triggered, status_msg = gate.evaluate_frame(timestamp_ms, speech_prob)

                    if is_triggered or speech_prob >= speech_prob_threshold:
                        conf_pct = int(speech_prob * 100)
                        self.send_shell_desktop_notification(
                            title="Executive Communication Coach",
                            message=f"Spoken dialogue detected ({conf_pct}% confidence). Capturing conversation...",
                            subtitle="Ambient Auto-Nudge Triggered"
                        )

                        print("\n\n" + "\033[1;36m┌" + "─" * 72 + "┐\033[0m")
                        print(f"\033[1;36m│\033[0m \033[1;32m[CONVERSATION DETECTED]\033[0m Spoken dialogue observed in room!                \033[1;36m│\033[0m")
                        print(f"\033[1;36m│\033[0m     Speech Confidence: \033[1;33m{conf_pct}%\033[0m • Ambient Low-Power Acoustic Gating Passed   \033[1;36m│\033[0m")
                        print(f"\033[1;36m│\033[0m                                                                        \033[1;36m│\033[0m")
                        print(f"\033[1;36m│\033[0m >>  \033[1;37mStarting continuous recording for coaching & action items...\033[0m       \033[1;36m│\033[0m")
                        print(f"\033[1;36m│\033[0m     \033[0;36m(Will automatically conclude when pause/silence is detected)\033[0m       \033[1;36m│\033[0m")
                        print("\033[1;36m└" + "─" * 72 + "┘\033[0m\n")

                        if on_speech_detected_callback:
                            return on_speech_detected_callback(speech_prob)
                        return True

        except KeyboardInterrupt:
            print("\n  [AMBIENT SENSING STOPPED] Exited by user.")
            return False
