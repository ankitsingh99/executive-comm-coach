# Executive Conversational Intelligence & Communication Coach

An on-device conversational intelligence platform and executive communication coach for professionals. Built for Google Pixel hardware and Android 14/15/16, it provides structured, persona-calibrated executive feedback (Executive Brevity / BLUF, Assertiveness, Active Listening, Hinglish Polish, and Candor) with strict DPDP Act (2023) privacy-by-design compliance.

---

## System Architecture

```
+---------------------------------------------------------------------------------------+
|                                STAGED DATAFLOW ARCHITECTURE                           |
+---------------------------------------------------------------------------------------+
| [Stage 1: Ambient VAD]                                                                |
| AudioRecord API (16kHz PCM) ---> Silero VAD (ONNX Runtime, <1ms latency)             |
|                                                                                       |
| [Stage 2: User-Consented Capture]                                                     |
| User Click ---> FGS Microphone Capture ---> Local Storage (Opus 24kbps AES-256)       |
|                                                                                       |
| [Stage 3: ASR & Diarization Engine]                                                   |
| Audio Stream ---> Sarvam AI Saaras v3 / Whisper ---> Time-Stamped Hinglish Turns      |
|                                                                                       |
| [Stage 4: Persona Context Injection]                                                  |
| Dynamic Dialog UI ---> Metadata Binding: {Speaker, Power Axis, Relational Rubric}     |
|                                                                                       |
| [Stage 5: LLM Executive Coaching Engine]                                              |
| Combined Context ---> Gemini Nano (AICore) / LiteRT-LM ---> Top-N Structured Feedback |
+---------------------------------------------------------------------------------------+
```

---

## Repository Directory Structure

```
executive-comm-coach/
├── core/                                  # Core Intelligence & Pipeline MVP
│   ├── engine/
│   │   ├── schema.py                      # Deterministic Pydantic / JSON Coaching Schemas
│   │   ├── persona_ontology.py            # Power Axis Strategies (Upward, Lateral, Downward)
│   │   ├── metrics_calculator.py          # Presence, Assertiveness, Filler words, Pause analysis
│   │   ├── local_coaching_synthesizer.py  # Pure on-device semantic coaching synthesizer
│   │   └── coaching_engine.py             # LLM Executive Coaching Prompter & Evaluator
│   ├── asr_diarization/
│   │   ├── vad_gater.py                   # Silero VAD gating & threshold simulation (tau >= 0.75)
│   │   ├── local_stt_engine.py            # On-device Hinglish STT & diarizer
│   │   ├── sarvam_client.py               # Sarvam AI Saaras v3 client for Hinglish STT
│   │   └── diarizer.py                    # Speaker separation ([USER] vs [COUNTERPART])
│   ├── privacy/
│   │   ├── pii_redactor.py                # Local NER sensitive data & secret scrubber
│   │   └── dpdp_compliance.py             # Statutory notice, consent, chime, & Right to Erasure
│   ├── tests/
│   │   ├── test_coaching_engine.py        # Validates exact Top-N schema, score ranges, quotes
│   │   ├── test_persona_ontology.py       # Upward (BLUF) vs Lateral vs Downward rubrics
│   │   ├── test_asr_and_redaction.py      # Hinglish transcript handling & PII wipe
│   │   └── test_dpdp_compliance.py        # Statutory consent logging & erasure
│   ├── config.py                          # Local processing mode configuration
│   ├── cli_coach.py                       # Interactive on-device CLI coach
│   ├── demo_live.py                       # Step-by-step live demonstration
│   └── run_mvp_demo.py                    # End-to-end interactive runner
│
├── android/                               # Android Application (Jetpack Compose + Architecture)
│   ├── app/src/main/
│   │   ├── AndroidManifest.xml            # ForegroundService type="microphone", permissions
│   │   └── java/com/execcoach/
│   │       ├── service/
│   │       │   ├── AmbientAudioService.kt # ForegroundService microphone lifecycle
│   │       │   ├── SileroVadDetector.kt   # ONNX Runtime Mobile VAD detector
│   │       │   └── AudioRecordManager.kt  # 16kHz PCM ring buffer & AES-256 Opus writer
│   │       ├── data/local/                # Room + SQLCipher hardware AES-256 persistence
│   │       ├── ui/
│   │       │   ├── dashboard/             # Executive KPI metrics & session history
│   │       │   ├── session/               # Instant post-meeting feedback bottom sheet
│   │       │   ├── overlay/               # Floating action bubble overlay
│   │       │   └── theme/                 # Material 3 Executive Dark/Light theme
│   │       ├── MainActivity.kt
│   │       └── ExecCoachApplication.kt
│   └── build.gradle.kts
└── README.md
```

---

## Running the Core MVP Pipeline

### 1. Execute the Live Step-by-Step Demonstration
```bash
python core/demo_live.py
```

### 2. Execute the Interactive Local CLI Coach
```bash
python core/cli_coach.py
```

### 3. Run the Automated Test Suite
```bash
PYTHONPATH=core pytest core/tests
```

---

## DPDP Act (2023) Privacy-by-Design Features

1. **Audible Notification Chime**: Distinct acoustic cue informing all participants before recording.
2. **Local Cryptographic Isolation**: Audio recordings and transcripts are encrypted with AES-256 via SQLCipher.
3. **Local PII & Secret Redaction**: Scans and masks phone numbers, PAN, Aadhaar, salary numbers, internal endpoints, and API credentials before sending to LLM runtimes.
4. **Statutory Right to Erasure**: Permanent one-touch purge of all audio chunks, transcript indices, and derived evaluation metrics.
