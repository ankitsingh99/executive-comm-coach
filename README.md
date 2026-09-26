# Executive Communication Coach

[![CI](https://github.com/ankitsingh99/executive-comm-coach/actions/workflows/ci.yml/badge.svg)](https://github.com/ankitsingh99/executive-comm-coach/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/ankitsingh99/executive-comm-coach?color=blue&label=Release)](https://github.com/ankitsingh99/executive-comm-coach/releases)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-3776AB?logo=python&logoColor=white)](https://github.com/ankitsingh99/executive-comm-coach)
[![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Linux%20%7C%20Android-lightgrey.svg)](https://github.com/ankitsingh99/executive-comm-coach)
[![Coverage](https://img.shields.io/badge/Coverage-95.83%25-brightgreen.svg)](https://github.com/ankitsingh99/executive-comm-coach)
[![Discussions](https://img.shields.io/badge/Discussions-Join%20Community-purple?logo=github)](https://github.com/ankitsingh99/executive-comm-coach/discussions)
[![Privacy](https://img.shields.io/badge/Privacy-DPDP%20Act%20Compliant-10B981.svg)](https://github.com/ankitsingh99/executive-comm-coach)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An on-device, privacy-first AI communication intelligence system and companion service that analyzes spoken workplace and personal conversations across all registers (Executive, Collaborative, Casual, Solo Practice, Mentorship, and Conflict), providing persona-calibrated communication coaching, cross-talk detection, commitment tracking, real-time acoustic waveform visualization, and a dedicated **Android companion app** optimized for modern flagship devices (including Google Pixel 11 on Android 17).

For an in-depth architectural breakdown and component diagram, see [ARCHITECTURE.md](ARCHITECTURE.md). For module and REST API specifications, see [API_REFERENCE.md](API_REFERENCE.md).

---

## 1. Core Capabilities

- **Universal Communication Coaching**: Automatically adapts coaching rubrics across 6 registers:
  - **Formal / Executive (UPWARD)**: Bottom-Line-Up-Front (BLUF) synthesis, quantified business impact, and proactive decisions.
  - **Collaborative / Peer (LATERAL)**: Shared milestone alignment, dependency tracking, and reciprocity.
  - **Casual / Social (CASUAL)**: Natural conversational cadence, warmth, and engagement flow.
  - **Solo Practice (SOLO)**: Monologue enunciation, structured inquiry, and thesis testing.
  - **Mentorship (DOWNWARD)**: Socratic questions, constructive guidance, and psychological safety.
  - **Difficult / Conflict Resolution (CONFLICT)**: Objective de-escalation, mutual resolution criteria, and neutral framing.
- **Multilingual & Hinglish Code-Mixing Support**: Native comprehension of Hindi, Indian English, and code-mixed Hinglish with South Asian discourse particles (*matlab*, *yaani*, *haina*, *arre*, *bhai*, *theek hai*).
- **Phonetic Hesitation & Non-Verbal Sound Detection**: Accurately itemizes vocal elongations (*ummm*, *aaaa*, *hmmm*), tongue clicks/tut-tuts (*tch*, *tsk*, *tch-tch*), and sigh sounds (*uff*, *oof*).
- **Real-Time Microphone Waveform Oscilloscope & Pitch Tracker**: Live Web Audio API `AnalyserNode` time-domain oscilloscope, real RMS energy tracking, dynamic visual gain boost, and fundamental vocal pitch estimation ($85\text{ Hz} - 450\text{ Hz}$).
- **Overlapping Speech & Cross-Talk Diarization**: Detects simultaneous speech turns, tags who interrupted whom, computes overlap duration, and provides coaching on floor-holding and intentional turn-taking.
- **Automated Commitments & Action Items Engine**: Automatically captures follow-up calls, deliverables, scheduling promises, and deadlines with smart AM/PM inference and date resolution.
- **Acoustic Voiceprint Memory Vault**: On-device biometric voiceprint recognition using local acoustic features with DPDP-compliant consent prompts and Section 12 Right-to-Erasure.
- **Native Android App (Google Pixel & Android 17 Optimized)**: Hardware-accelerated edge-to-edge container, display cutout safe area insets, runtime microphone permission handling, and offline asset embedding.

---

## 2. Quickstart Guide

### Prerequisites
- **Backend / Web**: Python 3.10+ (macOS Apple Silicon or Linux) with microphone access
- **Android App**: Android 10.0+ (API 29+) through Android 17+ (API 35+)

---

### Python Backend & Web Studio Setup

#### Local Installation (Editable Developer Mode)
```bash
# Clone the repository
git clone https://github.com/ankitsingh99/executive-comm-coach.git
cd executive-comm-coach

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install package in editable mode
pip install -e .
```

#### Run Live Microphone Coaching (Terminal CLI)
```bash
# Run live microphone coaching (records dynamically until silence after speech):
./record.sh

# Adjust pause/silence threshold:
./record.sh --silence 1.5

# Specify communication mode / context:
./record.sh --axis UPWARD --counterpart "Director"
./record.sh --axis LATERAL --counterpart "Colleague"
./record.sh --axis SOLO
```
*Tip: Press <kbd>Enter</kbd> or <kbd>Ctrl+C</kbd> at any time to finish speaking immediately.*

#### Launch Interactive App Studio & HUD (Web Browser)
```bash
# Start local server on port 8080:
python core/server.py 8080
# Or using the installed CLI entrypoint:
comm-coach-server 8080

# Open in browser:
open http://localhost:8080
```

#### Run Ambient Conversation Monitor (Nudge)
```bash
# Passively monitors room for conversation onset with < 2.5% CPU:
./nudge.sh
```

#### Run Automated Test Suite
```bash
./venv/bin/pytest -v
```
*55 passing unit and integration tests covering ASR, diarization, VAD gating, Indic normalization, temporal resolution, and coaching synthesis.*

---

## 3. Android Companion App (Google Pixel / Android 17)

The Android companion application provides the full-featured **Executive Communication Coach** cockpit directly on your Android smartphone with native microphone capture, real-time waveform visualization, scenario simulation, and offline operation.

### Assembling the Debug APK
```bash
# Navigate to the android/ directory
cd android

# Compile the debug APK
./gradlew assembleDebug
```
The compiled APK will be located at:
```
android/app/build/outputs/apk/debug/app-debug.apk
```

---

### Installation on Google Pixel Devices

#### Method 1: Instant Install via ADB (USB Cable)
1. **Enable Developer Options**: Open phone **Settings** $\rightarrow$ **About phone** $\rightarrow$ tap **Build number** 7 times.
2. **Enable USB Debugging**: Go to **Settings** $\rightarrow$ **System** $\rightarrow$ **Developer options** $\rightarrow$ turn on **USB debugging**.
3. **Connect Device**: Plug your Pixel into your Mac/PC via USB-C cable and tap **Allow** on the phone prompt.
4. **Install APK**:
   ```bash
   adb install -r android/app/build/outputs/apk/debug/app-debug.apk
   ```

#### Method 2: Wireless Debugging (No Cable)
1. Connect your Pixel to the same Wi-Fi network as your computer.
2. Go to **Settings** $\rightarrow$ **System** $\rightarrow$ **Developer options** $\rightarrow$ **Wireless debugging** $\rightarrow$ enable it.
3. Tap **Pair device with pairing code** (note IP, port, and 6-digit code).
4. Run:
   ```bash
   adb pair <IP>:<PAIR_PORT>
   # Enter the 6-digit pairing code when prompted
   adb connect <IP>:<CONNECT_PORT>
   adb install -r android/app/build/outputs/apk/debug/app-debug.apk
   ```

#### Method 3: Direct Sideloading (Files by Google)
If transferring the APK file via Google Drive, Gmail, or USB storage:
1. Open the built-in **Files by Google** app on your Pixel.
2. Tap **Apps** $\rightarrow$ select the **App install files (.apk)** tab.
3. Tap `app-debug.apk`.
4. When prompted: *"Your phone currently isn't allowed to install unknown apps from this source"*, tap **Settings** $\rightarrow$ toggle **Allow from this source** to **ON** $\rightarrow$ tap **Install**.

> [!TIP]
> **Fix for "Unsupported file" error on Pixel**: Do not tap the APK directly inside Google Drive or Chrome's download bar (which invokes the Docs/Drive viewer). Always open the APK through the native **Files by Google** app under **App install files (.apk)**.

---

## 4. Modular Architecture Overview

```
executive-comm-coach/
├── android/                          # Native Android companion application
│   ├── app/
│   │   ├── src/main/
│   │   │   ├── assets/               # Embedded offline coaching studio & HUD
│   │   │   │   └── index.html
│   │   │   ├── java/com/execcoach/
│   │   │   │   ├── MainActivity.kt   # Edge-to-edge hardware-accelerated container & mic permissions
│   │   │   │   ├── ExecCoachApplication.kt
│   │   │   │   ├── service/          # Ambient audio & Silero VAD services
│   │   │   │   └── data/local/       # Encrypted SQLCipher database & Room DAOs
│   │   │   └── AndroidManifest.xml   # Permissions (RECORD_AUDIO, FOREGROUND_SERVICE)
│   │   └── build.gradle.kts          # Gradle build configuration (API 35/36/37)
│   └── gradlew                       # Gradle build wrapper
│
├── core/
│   ├── asr_diarization/              # Audio streaming, VAD gating, STT & voiceprints
│   │   ├── acoustic_speaker_detector.py # Pitch, RMS energy, and vocal tone classifier
│   │   ├── diarizer.py                  # Speaker turn alignment & overlap detection
│   │   ├── gemini_audio_engine.py       # Gemini multimodal transcription & tone sensing
│   │   ├── indic_normalizer.py          # Devanagari transliteration & Hinglish harmonization
│   │   ├── live_mic_recorder.py         # Native sounddevice / CoreAudio real-time streamer
│   │   ├── local_stt_engine.py          # Faster-Whisper on-device STT fallback
│   │   ├── nvidia_parakeet_engine.py    # Local CTC/RNNT transducer STT integration
│   │   ├── sarvam_client.py             # Cloud Saarathi/Sarvam Indic speech client
│   │   ├── speaker_voiceprint_registry.py # Persistent acoustic biometric vault
│   │   └── vad_gater.py                 # Silero-style low-power ambient acoustic gate
│   │
│   ├── engine/                       # Linguistic analysis, metrics & coaching
│   │   ├── action_item_extractor.py     # Extracts commitments, follow-ups & deadlines
│   │   ├── coaching_engine.py           # Master coaching facade (Gemini + local fallback)
│   │   ├── gemini_coaching_engine.py    # Deep semantic coaching via Gemini
│   │   ├── local_coaching_synthesizer.py # 100% on-device offline NLP coaching engine
│   │   ├── metrics_calculator.py        # Presence, assertiveness & listening scoring
│   │   ├── persona_ontology.py          # Multi-register communication rubrics (6 modes)
│   │   ├── schema.py                    # Strict Pydantic & dataclass schemas
│   │   ├── temporal_resolver.py         # Smart date/time & AM/PM resolver
│   │   └── transcription_analyzer.py    # Highlights, metrics & task pipeline
│   │
│   ├── privacy/                      # Statutory DPDP compliance & PII redaction
│   │   ├── dpdp_compliance.py           # Statutory consent, chime & right-to-erasure
│   │   └── pii_redactor.py              # Pattern-based entity & credential scrubber
│   │
│   ├── config.py                     # Configuration, API keys, and model paths
│   ├── record_live_coach.py          # Live interactive hardware microphone coach
│   ├── cli_coach.py                  # Offline transcript coaching CLI
│   └── server.py                     # REST & WebSocket API server
│
├── emulator/
│   └── index.html                    # Glassmorphic interactive Web UI & mobile simulator
│
├── ARCHITECTURE.md                   # Complete architectural reference & data contracts
├── API_REFERENCE.md                  # Comprehensive module & REST API manual
├── CHANGELOG.md                      # Chronological version changelog
├── RELEASE_NOTES.md                  # Curated release notes
├── record.sh                         # Live microphone execution script
└── nudge.sh                          # Ambient conversation monitor script
```

---

## 5. Documentation & Developer Guides

- **[System Architecture Guide](ARCHITECTURE.md)**: End-to-end dataflow diagrams, pipeline contracts, and Android container design.
- **[API Reference & Manual](API_REFERENCE.md)**: Exhaustive class, method, data structure, and REST API documentation.
- **[Changelog](CHANGELOG.md)**: Complete chronological history of updates and improvements.
- **[Release Notes](RELEASE_NOTES.md)**: Highlights of current and past feature releases.

---

## 6. Disclaimer & AI Attribution

> [!NOTE]
> **AI Assistance Notice**: This software tool, including its core speech processing algorithms, NLP coaching logic, test suites, architecture, and user interface, was created and developed with the assistance of Artificial Intelligence (AI) systems and models. All coaching insights, transcription analysis, and behavioral scores are generated algorithmically for developmental and self-improvement purposes.

---

## 7. License

Licensed under the [Apache License, Version 2.0](LICENSE).
