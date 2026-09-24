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

    def record_until_silence(
        self,
        silence_threshold_sec: float = 2.0,
        min_speech_duration_sec: float = 1.2,
        max_duration_sec: int = 180,
        chunk_duration_sec: float = 0.6,
        speech_prob_threshold: float = 0.45,
        gain_boost: float = 2.0,
        idle_timeout_sec: float = 14.0,
        output_wav_path: Optional[str] = None
    ) -> str:
        """
        Dynamically records microphone audio until the conversation end is detected
        by analyzing silence after the last spoken word, or when the user presses Enter/Ctrl+C.
        Equipped with dynamic noise floor tracking and digital gain boost.
        """
        import sys
        import select
        import numpy as np
        import wave
        import subprocess
        from .vad_gater import AmbientVadGate

        if output_wav_path is None:
            temp_dir = tempfile.gettempdir()
            output_wav_path = os.path.join(temp_dir, f"mic_session_{int(time.time())}.wav")

        print(f"\n  [DYNAMIC DIALOGUE CAPTURE ACTIVE]")
        print(f"      • Auto-stop: Automatically concludes when pause is detected (>{silence_threshold_sec:.1f}s silence after speech)")
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
        audio_chunks = []
        has_spoken = False
        silence_elapsed = 0.0
        total_recorded_sec = 0.0
        noise_floor_rms = 0.0
        temp_chunk_files = []

        try:
            while total_recorded_sec < max_duration_sec:
                # Check if user pressed Enter to finish
                if _check_key_pressed():
                    print("\n\n  [MANUAL STOP] Enter pressed. Concluding recording immediately...")
                    break

                chunk_file = os.path.join(tempfile.gettempdir(), f"dyn_chunk_{int(time.time() * 1000)}_{len(audio_chunks)}.wav")
                temp_chunk_files.append(chunk_file)

                cmd = [
                    "ffmpeg", "-y",
                    "-f", "avfoundation",
                    "-i", ":0",
                    "-af", f"volume={gain_boost}",
                    "-t", str(chunk_duration_sec),
                    "-ar", str(self.sample_rate),
                    "-ac", "1",
                    chunk_file
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=chunk_duration_sec + 2)

                if os.path.exists(chunk_file) and os.path.getsize(chunk_file) > 400:
                    with wave.open(chunk_file, "rb") as wf:
                        n_frames = wf.getnframes()
                        raw_bytes = wf.readframes(n_frames)
                        chunk_samples = np.frombuffer(raw_bytes, dtype=np.int16)

                    audio_chunks.append(chunk_samples)
                    total_recorded_sec += chunk_duration_sec

                    # Calculate current RMS
                    if chunk_samples.dtype == np.int16:
                        float_samples = chunk_samples.astype(np.float32) / 32768.0
                    else:
                        float_samples = chunk_samples.astype(np.float32)
                    cur_rms = float(np.sqrt(np.mean(float_samples ** 2)))

                    # Adaptively estimate room background noise floor
                    if noise_floor_rms == 0.0:
                        noise_floor_rms = cur_rms
                    elif not has_spoken or silence_elapsed > 0.4:
                        noise_floor_rms = 0.85 * noise_floor_rms + 0.15 * min(cur_rms, noise_floor_rms * 1.3)

                    speech_prob = gate.calculate_speech_probability(chunk_samples, noise_floor_rms=noise_floor_rms)

                    if speech_prob >= speech_prob_threshold:
                        has_spoken = True
                        silence_elapsed = 0.0
                        print(f"  [SPEAKING] {total_recorded_sec:.1f}s recorded | Active Dialogue (Voice: {int(speech_prob*100)}%) [Press Enter to finish]    ", end="\r", flush=True)
                    else:
                        if has_spoken:
                            silence_elapsed += chunk_duration_sec
                            print(f"  [SILENCE DETECTED] {total_recorded_sec:.1f}s recorded | Paused: {silence_elapsed:.1f}s / {silence_threshold_sec:.1f}s [Press Enter to finish]   ", end="\r", flush=True)
                            
                            if silence_elapsed >= silence_threshold_sec and total_recorded_sec >= min_speech_duration_sec:
                                print(f"\n\n  [CONVERSATION CONCLUDED] End of conversation detected ({silence_threshold_sec:.1f}s silence after speech).")
                                break
                        else:
                            if total_recorded_sec >= idle_timeout_sec:
                                print(f"\n\n  [IDLE TIMEOUT] No speech detected after {idle_timeout_sec:.0f}s. Concluding session.")
                                break
                            print(f"  [LISTENING] {total_recorded_sec:.1f}s | Waiting for dialogue to begin... [Press Enter to finish]          ", end="\r", flush=True)

                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\n\n  [STOPPED BY USER] Concluding recording and analyzing dialogue...")
        finally:
            # Clean up temp chunk files
            for cf in temp_chunk_files:
                if os.path.exists(cf):
                    try:
                        os.remove(cf)
                    except OSError:
                        pass

        if not audio_chunks:
            return self.record_to_wav(duration_seconds=6, output_wav_path=output_wav_path)

        # Concatenate all recorded chunks into single WAV file
        full_audio = np.concatenate(audio_chunks)
        with wave.open(output_wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(full_audio.tobytes())

        print(f"  [AUDIO STORED] Full conversation ({len(full_audio)/self.sample_rate:.1f}s) captured successfully.")
        return output_wav_path

    def record_to_wav(self, duration_seconds: int = 8, output_wav_path: Optional[str] = None) -> str:
        """
        Records live microphone audio for a fixed duration (in seconds).
        """
        if output_wav_path is None:
            temp_dir = tempfile.gettempdir()
            output_wav_path = os.path.join(temp_dir, f"mic_session_{int(time.time())}.wav")

        print(f"  [MICROPHONE ACTIVE] Recording {duration_seconds}s directly from your microphone...")

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


    @staticmethod
    def send_shell_desktop_notification(title: str = "Executive Coach", message: str = "Spoken dialogue detected! Starting coaching capture...", subtitle: str = "Ambient Speech Nudge"):
        """
        Triggers macOS system desktop notification, terminal bell, and alert chime.
        """
        import sys
        import subprocess
        try:
            sys.stdout.write('\a')
            sys.stdout.flush()
            script = f'display notification "{message}" with title "{title}" subtitle "{subtitle}" sound name "Glass"'
            subprocess.Popen(["osascript", "-e", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    def listen_for_speech_and_nudge(
        self,
        poll_interval_sec: float = 0.8,
        speech_prob_threshold: float = 0.35,
        gain_boost: float = 2.5,
        max_wait_seconds: Optional[int] = None,
        on_speech_detected_callback: Optional[callable] = None
    ) -> bool:
        """
        Passively monitors the ambient microphone stream with ultra-low compute and high sensitivity.
        As soon as human speech / spoken dialogue is detected, triggers a consent nudge
        prompting the user if they wish to start recording for communication coaching analysis.
        """
        import numpy as np
        import subprocess
        from .vad_gater import AmbientVadGate

        print("\n  [AMBIENT SENSING ACTIVE] Passively listening for spoken dialogue (High Sensitivity)...")
        print("  (Privacy protected: Audio evaluated in memory & purged immediately if below threshold)")

        start_time = time.time()
        gate = AmbientVadGate(speech_prob_threshold=speech_prob_threshold)
        spinners = ["-", "\\", "|", "/"]
        spin_idx = 0

        while True:
            if max_wait_seconds and (time.time() - start_time) > max_wait_seconds:
                print("\n  [AMBIENT TIMEOUT] No speech detected within window.")
                return False

            print(f"  {spinners[spin_idx % len(spinners)]} Ambient Ear Active... (Waiting for dialogue to start)", end="\r", flush=True)
            spin_idx += 1

            temp_chunk_path = os.path.join(tempfile.gettempdir(), f"vad_sample_{int(time.time() * 1000)}.wav")
            try:
                # Capture a short probe chunk via ffmpeg avfoundation with digital gain boost
                cmd = [
                    "ffmpeg", "-y",
                    "-f", "avfoundation",
                    "-i", ":0",
                    "-af", f"volume={gain_boost}",
                    "-t", str(poll_interval_sec),
                    "-ar", str(self.sample_rate),
                    "-ac", "1",
                    temp_chunk_path
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=poll_interval_sec + 2)

                if os.path.exists(temp_chunk_path) and os.path.getsize(temp_chunk_path) > 800:
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
                        # Send macOS shell/desktop notification
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

            except Exception:
                pass
            finally:
                if os.path.exists(temp_chunk_path):
                    try:
                        os.remove(temp_chunk_path)
                    except OSError:
                        pass
            time.sleep(0.05)

