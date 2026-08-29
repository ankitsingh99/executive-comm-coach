# Executive Communication Coach

An on-device, privacy-first AI communication intelligence system and Android service that analyzes spoken workplace conversations and provides persona-calibrated executive coaching.

---

## 1. System Architecture

```
[Hardware Microphone / Ambient Audio Service]
                      │
                      ▼
[Silero VAD Acoustic Gate (600ms trigger / 3s silence purge)]
                      │
                      ▼
[DPDP Statutory Consent & Audible Notification Chime]
                      │
                      ▼
[NVIDIA Parakeet STT (nvidia/parakeet-ctc-0.6b Conformer-CTC)]
                      │
                      ▼
[PII & Confidential Entity Redactor (Regex / Tokenization)]
                      │
                      ▼
[Semantic Intent & Relational Coaching Synthesizer]
  ├── Intent Classification (Question, Status, Proposal, Blocker)
  ├── Topic & Noun Extraction
  ├── Quantitative Scoring (Presence, Assertiveness, Listening)
  └── Dynamic Persona Calibration (Upward BLUF, Lateral, Downward)
                      │
                      ▼
[Executive Scorecard & Bespoke Coached BLUF Alternatives]
                      │
                      ▼
[SQLCipher / DPDP Section 12 Right-to-Erasure Pipeline]
```

---

- **Interactive Web & Mobile App Emulator**: Glassmorphic real-time coaching interface with live browser microphone capture, speech recognition, waveform visualizer, comprehensive feedback insights drawer, and interactive voice vault.
- **NVIDIA Parakeet & Google Gemini STT**: Multimodal audio transcription and tone sensing running locally or via Gemini Live APIs.
- **Dynamic Semantic Intent & BLUF Coaching**: Automatically transforms passive, hypothetical statements into decisive, proactive Bottom-Line-Up-Front (BLUF) executive assertions.
- **🇮🇳 Multilingual & Hinglish Support**: Code-mixed Hindi/English comprehension, hesitation markers (*matlab*, *yaani*, *haina*), hedging detection, and action item temporal parsing (*"kal 10 baje"*, *"shaam tak"*).
- **Automated Commitments & Action Items Engine**: Automatically captures promises, follow-up calls, deadlines, and deliverables from spoken conversations.
- **Persistent Biometric Voiceprints**: On-device voiceprint memory vault recognizing speakers across conversations with DPDP-compliant consent prompts.
- **Relational Persona Ontology**: Calibrated against three organizational power axes:
  - **Upward (Executive / Leadership)**: BLUF synthesis, quantified business impact, decisive recommendations.
  - **Lateral (Peer / Product)**: Collaborative framing, mutual benefit, shared dependency alignment.
  - **Downward (Direct Report / Mentee)**: Psychological safety, Socratic questioning, developmental inquiry.
- **DPDP Act 2023 Compliance**: Privacy-by-design architecture featuring statutory audible chime notifications, automated PII scrubbing (PAN, Aadhaar, secrets, financial figures), and Section 12 Right-to-Erasure purging.
- **Android Native Architecture**: Android 14/15/16 Foreground Service with ONNX Runtime Mobile Silero VAD, 16kHz ring buffer, AES-256 Opus encryption, Room/SQLCipher encrypted storage, and Dagger Hilt.

---

## 3. Quickstart Guide

### Prerequisites
- Python 3.10+ (macOS Apple Silicon or Linux)
- Microphone access

### Setup Environment
```bash
# Clone the repository
git clone https://github.com/ankitsingh99/executive-comm-coach.git
cd executive-comm-coach

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Launch Interactive App Emulator (Web & Mobile)
```bash
# Start local emulator server on port 8080:
python core/server.py 8080

# Open in browser:
open http://localhost:8080
```

### Run Live Terminal Microphone Coaching
```bash
# Run live microphone coaching (records from hardware microphone):
./record.sh 8

# Specify custom duration and power axis:
./record.sh 15 --axis UPWARD --counterpart "Director" --role "Engineering Director"
./record.sh 10 --axis LATERAL --counterpart "Colleague" --role "Product Lead"
./record.sh 10 --axis DOWNWARD --counterpart "Team Member" --role "Associate Engineer"
```

### Run Automated Unit Tests
```bash
./venv/bin/pytest -v
```

---

## 4. Repository Structure

```
executive-comm-coach/
├── core/
│   ├── server.py                     # HTTP server powering the interactive app emulator
│   ├── asr_diarization/
│   │   ├── diarizer.py               # Diarization & verbal self-introduction extraction
│   │   ├── speaker_voiceprint_registry.py # Persistent acoustic biometric vault
│   │   ├── live_mic_recorder.py      # CoreAudio / sounddevice 16kHz PCM capture
│   │   ├── gemini_audio_engine.py    # Google Gemini audio & tone sensing
│   │   ├── nvidia_parakeet_engine.py # NVIDIA Parakeet CTC STT inference
│   │   ├── local_stt_engine.py       # Speech recognition dispatcher
│   │   ├── vad_gater.py              # Silero VAD acoustic filter
│   │   ├── acoustic_speaker_detector.py # Pitch & vocal cadence detection
│   │   └── sarvam_client.py          # Multilingual diarization adapter
│   ├── engine/
│   │   ├── action_item_extractor.py  # Commitment, deadline & action items extractor
│   │   ├── coaching_engine.py        # Master executive coaching pipeline
│   │   ├── local_coaching_synthesizer.py # Local deterministic BLUF synthesizer
│   │   ├── gemini_coaching_engine.py # Gemini GenAI coaching synthesizer
│   │   ├── metrics_calculator.py     # Presence, assertiveness, Hinglish fillers
│   │   ├── persona_ontology.py       # Upward/Lateral/Downward persona models
│   │   └── schema.py                 # Structured dataclasses and JSON models
│   ├── privacy/
│   │   ├── dpdp_compliance.py        # DPDP consent & right-to-erasure
│   │   └── pii_redactor.py           # Aadhaar, PAN, email, phone scrubbing
│   └── tests/                        # 30 automated unit tests (100% passing)
├── emulator/
│   └── index.html                    # Glassmorphic interactive mobile app emulator
└── android/                          # Native Android 15/16 Jetpack Compose project
    ├── app/src/main/                 # Kotlin Compose UI, Foreground Service & Room DB
```

---

## 5. License

Licensed under the Apache License, Version 2.0.
