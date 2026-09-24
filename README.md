# Executive Communication Coach

An on-device, privacy-first AI communication intelligence system and companion service that analyzes spoken workplace and personal conversations across all registers (Executive, Collaborative, Casual, Solo Practice, Mentorship, and Conflict), providing persona-calibrated communication coaching, cross-talk detection, and commitment tracking.

For an in-depth architectural breakdown and component diagram, see [ARCHITECTURE.md](file:///Users/ashish/git/executive-comm-coach/ARCHITECTURE.md).

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
- **Overlapping Speech & Cross-Talk Diarization**: Detects simultaneous speech turns, tags who interrupted whom, computes overlap duration, and provides coaching on floor-holding and intentional turn-taking.
- **Automated Commitments & Action Items Engine**: Automatically captures follow-up calls, deliverables, scheduling promises, and deadlines with smart AM/PM inference and date resolution.
- **Acoustic Voiceprint Memory Vault**: On-device biometric voiceprint recognition using local acoustic features with DPDP-compliant consent prompts and Section 12 Right-to-Erasure.
- **Interactive Web & Mobile App Emulator**: Modern glassmorphic real-time coaching interface with live browser microphone capture, speech recognition, waveform visualizer, and insights drawer.

---

## 2. Quickstart Guide

### Prerequisites
- Python 3.10+ (macOS Apple Silicon or Linux)
- Microphone access

### Installation & Packaging

#### Option A: Install directly from Git
```bash
pip install git+https://github.com/ankitsingh99/executive-comm-coach.git
```
Once installed, run the coach from any terminal window:
```bash
executive-comm-coach --axis UPWARD
comm-coach-server 8080
```

#### Option B: Developer Setup (Local Editable Mode)
```bash
# Clone the repository
git clone https://github.com/ankitsingh99/executive-comm-coach.git
cd executive-comm-coach

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install in editable mode
pip install -e .
```

### Run Live Terminal Microphone Coaching
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

### Launch Interactive App Emulator (Web & Mobile)
```bash
# Start local emulator server on port 8080:
python core/server.py 8080
# Or using the installed CLI entrypoint:
comm-coach-server 8080

# Open in browser:
open http://localhost:8080
```


### Run Ambient Conversation Monitor (Nudge)
```bash
# Passively monitors room for conversation onset with < 2.5% CPU:
./nudge.sh
```

### Run Automated Test Suite
```bash
./venv/bin/pytest -v
```
*55 passing unit and integration tests covering ASR, diarization, VAD gating, Indic normalization, temporal resolution, and coaching synthesis.*

---

## 3. Modular Architecture Overview

```
executive-comm-coach/
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
│   └── index.html                    # Glassmorphic interactive Web UI
│
├── ARCHITECTURE.md                   # Complete architectural reference & data contracts
├── API_REFERENCE.md                  # Comprehensive module & REST API manual
├── record.sh                         # Live microphone execution script
└── nudge.sh                          # Ambient conversation monitor script
```

---

## 4. Documentation & Developer Guides

- **[System Architecture Guide](ARCHITECTURE.md)**: End-to-end dataflow diagrams, pipeline contracts, and operational guidelines.
- **[API Reference & Manual](API_REFERENCE.md)**: Exhaustive class, method, data structure, and REST API documentation.

---

## 5. License

Licensed under the Apache License, Version 2.0.


