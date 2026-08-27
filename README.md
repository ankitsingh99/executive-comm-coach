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

## 2. Key Features

- **NVIDIA Parakeet Speech-to-Text**: High-accuracy, low-latency 600M Conformer-CTC acoustic transcription model running locally on Apple Silicon MPS / CPU.
- **Dynamic Semantic Intent & BLUF Coaching**: Automatically decomposes spoken inquiries and transforms passive, hypothetical statements into decisive, proactive Bottom-Line-Up-Front (BLUF) executive assertions.
- **Relational Persona Ontology**: Calibrated against three organizational power axes:
  - **Upward (Executive / Leadership)**: BLUF synthesis, quantified business impact, decisive recommendations.
  - **Lateral (Peer / Product)**: Collaborative framing, mutual benefit, shared dependency alignment.
  - **Downward (Direct Report / Mentee)**: Psychological safety, Socratic questioning, developmental inquiry.
- **DPDP Act 2023 Compliance**: Privacy-by-design architecture featuring statutory audible chime notifications, automated PII scrubbing (PAN, Aadhaar, secrets, financial figures), and Section 12 Right-to-Erasure purging.
- **Android Native Architecture**: Android 14/15/16 Foreground Service with ONNX Runtime Mobile Silero VAD, 16kHz ring buffer, AES-256 Opus encryption, and encrypted Room/SQLCipher storage.

---

## 3. Quickstart Guide

### Prerequisites
- Python 3.10+ (macOS Apple Silicon or Linux)
- Microphone access

### Setup Environment
```bash
# Clone the repository
git clone <repository_url>
cd executive-comm-coach

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run Live Microphone Coaching
```bash
# Run live microphone coaching (records 8s from hardware microphone):
./record.sh 8

# Specify custom duration and power axis:
./record.sh 15 --axis UPWARD --counterpart "Director" --role "Engineering Director"
./record.sh 10 --axis LATERAL --counterpart "Colleague" --role "Product Lead"
./record.sh 10 --axis DOWNWARD --counterpart "Team Member" --role "Associate Engineer"
```

### Run Automated Tests
```bash
./venv/bin/pytest
```

---

## 4. Repository Structure

```
executive-comm-coach/
├── core/
│   ├── asr_diarization/
│   │   ├── live_mic_recorder.py      # CoreAudio / sounddevice 16kHz PCM audio capture
│   │   ├── nvidia_parakeet_engine.py # NVIDIA Parakeet CTC STT inference engine
│   │   ├── local_stt_engine.py       # Speech recognition dispatcher (Parakeet + Whisper)
│   │   ├── vad_gater.py              # Silero VAD acoustic filter
│   │   ├── sarvam_client.py          # Indic bilingual Hinglish diarization client
│   │   └── diarizer.py               # Speaker turn segmentation
│   ├── engine/
│   │   ├── schema.py                 # Pydantic & dataclass schema models
│   │   ├── persona_ontology.py       # Relational power axis rubrics (Upward/Lateral/Downward)
│   │   ├── metrics_calculator.py     # Presence, Assertiveness, and Filler calculations
│   │   ├── local_coaching_synthesizer.py # Dynamic semantic intent & BLUF synthesizer
│   │   └── coaching_engine.py        # Evaluation facade
│   ├── privacy/
│   │   ├── pii_redactor.py           # Regex PII and secrets scrubber
│   │   └── dpdp_compliance.py        # DPDP Section 12 Right-to-Erasure & consent logging
│   ├── tests/                        # Unit test suite
│   ├── config.py                     # Processing configurations
│   └── record_live_coach.py          # Interactive hardware microphone CLI
├── android/                          # Native Android Jetpack Compose & Service subsystem
├── record.sh                         # Executable launcher script
├── requirements.txt                  # Python dependencies
├── pytest.ini                        # Pytest configuration
├── LICENSE                           # Apache 2.0 License
└── README.md                         # Project documentation
```

---

## 5. License

Licensed under the Apache License, Version 2.0.
