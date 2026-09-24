# Executive Communication Coach: API Reference & Developer Documentation

Welcome to the comprehensive API Reference and Developer Manual for the **Executive Communication Coach**. This document details every module, class, method, REST endpoint, and data contract in the codebase.

---

## Table of Contents
1. [Core Engine Package (`core.engine`)](#1-core-engine-package-coreengine)
   - [Schema & Data Contracts (`schema.py`)](#schemadata-contracts-schemapy)
   - [Metrics Calculator (`metrics_calculator.py`)](#metrics-calculator-metricscalculatorpy)
   - [Action Item & Temporal Resolver (`action_item_extractor.py`, `temporal_resolver.py`)](#action-item--temporal-resolver)
   - [Persona Ontology (`persona_ontology.py`)](#persona-ontology-personaontologypy)
   - [Coaching Engines (`coaching_engine.py`, `gemini_coaching_engine.py`, `local_coaching_synthesizer.py`)](#coaching-engines)
   - [Transcription Analyzer (`transcription_analyzer.py`)](#transcription-analyzer-transcriptionanalyzerpy)
2. [ASR & Diarization Package (`core.asr_diarization`)](#2-asr--diarization-package-coreasr_diarization)
   - [Live Mic Recorder (`live_mic_recorder.py`)](#live-mic-recorder-livemicrecorderpy)
   - [Speaker Voiceprint Registry (`speaker_voiceprint_registry.py`)](#speaker-voiceprint-registry-speakervoiceprintregistrypy)
   - [Diarization Engine (`diarizer.py`)](#diarization-engine-diarizerpy)
   - [Indic Normalizer (`indic_normalizer.py`)](#indic-normalizer-indicnormalizerpy)
   - [ASR Engines (`gemini_audio_engine.py`, `local_stt_engine.py`, `sarvam_client.py`)](#asr-engines)
3. [Privacy & Compliance Package (`core.privacy`)](#3-privacy--compliance-package-coreprivacy)
   - [PII Redactor (`pii_redactor.py`)](#pii-redactor-piiredactorpy)
   - [DPDP Compliance Manager (`dpdp_compliance.py`)](#dpdp-compliance-manager-dpdpcompliancepy)
4. [HTTP / WebSocket Server (`core.server`)](#4-http--websocket-server-coreserver)
   - [REST Endpoints](#rest-endpoints)
   - [Live WebSocket Streaming](#live-websocket-streaming)
5. [CLI & Entrypoint Scripts](#5-cli--entrypoint-scripts)

---

## 1. Core Engine Package (`core.engine`)

### Schema/Data Contracts (`schema.py`)

All engine components exchange immutable or strictly validated dataclasses:

#### `Utterance`
Represents a single conversational turn.
```python
@dataclass
class Utterance:
    speaker: str                      # E.g. "USER", "SPEAKER_01", "ASHISH"
    start_time: float                 # Seconds from recording onset
    end_time: float                   # Seconds from recording onset
    transcript: str                   # Redacted, normalized text of turn
    is_overlapping: bool = False      # True if speech overlaps with previous turn
    overlap_duration_sec: float = 0.0 # Duration in seconds of overlap
    interrupted_speaker: Optional[str] = None # Name of speaker interrupted, if any
```

#### `CommunicationMetrics`
Quantitative delivery and linguistic scorecard.
```python
@dataclass
class CommunicationMetrics:
    talk_time_ratio: float            # Ratio of user speech to total session speech [0.0 - 1.0]
    words_per_minute: float           # Speech cadence (WPM)
    presence_score: int               # Score bounded [15, 98]
    assertiveness_score: int          # Score bounded [15, 98]
    active_listening_score: int       # Score bounded [15, 98]
    vocal_tone_profile: str           # E.g. "Confident & Direct", "Collaborative"
    filler_words_detected: List[str]  # Detected fillers: "um", "matlab", "like", etc.
    interruption_count: int = 0       # Number of times user interrupted counterparts
    overlap_count: int = 0            # Number of cross-talk turn collisions
```

#### `ActionItem`
Actionable commitments and follow-ups with resolved deadlines.
```python
@dataclass
class ActionItem:
    owner: str                        # Task assignee (e.g. "User", "Priya", "Self")
    task: str                         # Concrete deliverable description
    due_time_or_date: str             # Extracted raw temporal string
    resolved_datetime: Optional[str]  # Absolute ISO timestamp (YYYY-MM-DD HH:MM)
    category: str                     # "DELIVERABLE", "MEETING", "FOLLOW_UP", "INVESTIGATION"
    urgency: str                      # "HIGH", "MEDIUM", "LOW"
```

---

### Metrics Calculator (`metrics_calculator.py`)

```python
from core.engine.metrics_calculator import MetricsCalculator

calculator = MetricsCalculator()
metrics: CommunicationMetrics = calculator.analyze_dialogue(
    utterances=dialogue_turns,
    user_speaker_tags=["USER", "ASHISH", "SELF"]
)
```

- **Dynamic Monologue Fallback**: If `user_speaker_tags` is not explicitly matched in a single-speaker session, `MetricsCalculator` automatically treats the solo speaker as the primary user to prevent empty scorecards.
- **Cross-talk & Interruption Scoring**: Analyzes turn start times against preceding turn end times with a `0.25s` buffer. Penalizes active listening score if user repeatedly starts speaking while counterpart is mid-sentence.

---

### Action Item & Temporal Resolver

#### `ActionItemExtractor` (`action_item_extractor.py`)
Extracts tasks using grammatical regex triggers across English and Hinglish (e.g., `"I will send"`, `"let me check"`, `"kal bhej dunga"`, `"we should schedule"`).

#### `TemporalResolver` (`temporal_resolver.py`)
Converts relative temporal expressions into concrete dates and ISO 8601 timestamps:
- `"kal sham 4 baje"` $\rightarrow$ Next day, 16:00
- `"by EOD"` $\rightarrow$ Current date, 18:00
- `"next Monday morning"` $\rightarrow$ Upcoming Monday, 09:00
- Resolves AM/PM ambiguities dynamically based on professional business hour heuristics.

---

### Persona Ontology (`persona_ontology.py`)

Configures 6 distinct executive registers:

| Register | Objective | Key Linguistic Dimensions |
| :--- | :--- | :--- |
| `UPWARD` (Executive) | Executive brevity & bottom-line first | High assertiveness, BLUF structure, zero hedging |
| `LATERAL` (Peer) | Cross-functional alignment | Collaborative consensus, clear handoffs |
| `DOWNWARD` (Mentorship) | Direction & psychological safety | Open-ended questioning, constructive framing |
| `SOLO` (Presentation) | Rhetorical clarity & pacing | Cadence control, vocal variety, structured signposting |
| `CASUAL` (Informal) | Rapport building | Warmth, active listening, authentic connection |
| `CONFLICT` (Negotiation) | De-escalation & objective boundaries | Neutral language, empathy anchoring, principled negotiation |

---

## 2. ASR & Diarization Package (`core.asr_diarization`)

### Live Mic Recorder (`live_mic_recorder.py`)

Native CoreAudio/ALSA streaming via `sounddevice`:
```python
from core.asr_diarization.live_mic_recorder import LiveMicRecorder

recorder = LiveMicRecorder(sample_rate=16000, channels=1)
wav_path = recorder.record_until_silence(
    silence_threshold_sec=1.5,
    max_duration_sec=300,
    gain_boost=1.35
)
```
- **Real-Time Visual VU Meter**: Displays instantaneous RMS voice activity percentage directly in the terminal.
- **Hybrid VAD Detection**: Combines energy threshold (`cur_rms >= 0.0055`) and WebRTC/Silero voice probability (`prob >= 0.32`).
- **Graceful Manual Termination**: Pressing <kbd>Enter</kbd> or <kbd>Ctrl+C</kbd> cleanly flushes in-memory audio buffers without process lockup.

### Speaker Voiceprint Registry (`speaker_voiceprint_registry.py`)

Provides local, persistent acoustic speaker memory:
- Extracts 13-dimensional MFCC vectors + spectral centroid, pitch, and delta features.
- Cosine-similarity thresholding ($\ge 0.82$) to recognize registered speakers across sessions.
- Fully compliant with DPDP Act: Supports single-command profile deletion (`registry.delete_voiceprint(speaker_name)`).

---

## 3. Privacy & Compliance Package (`core.privacy`)

### PII Redactor (`pii_redactor.py`)

Scrubs Personally Identifiable Information (PII) before any LLM dispatch or disk persistence:
- **Aadhaar Numbers**: `\d{4}\s\d{4}\s\d{4}` $\rightarrow$ `[REDACTED_AADHAAR]`
- **PAN Cards**: `[A-Z]{5}[0-9]{4}[A-Z]` $\rightarrow$ `[REDACTED_PAN]`
- **Indian Phone Numbers**: `(+91|0)?[6-9]\d{9}` $\rightarrow$ `[REDACTED_PHONE]`
- **Emails & Financial Data**: Credit cards, API tokens, and secret strings sanitized automatically.

---

## 4. HTTP / WebSocket Server (`core.server`)

Run the server on any port:
```bash
./venv/bin/python core/server.py 8080
```

### REST Endpoints

#### `POST /api/analyze-audio`
Upload a WAV/MP3 recording for transcription, diarization, and coaching.
- **Form Data**:
  - `file`: Audio binary (`audio/wav`)
  - `axis`: Power axis (`"UPWARD"`, `"LATERAL"`, etc.)
  - `counterpart`: Optional counterpart name
- **Response**: `ExecutiveCoachingEvaluation` JSON object.

#### `POST /api/analyze-transcript`
Submit raw text or diarized turns for immediate linguistic coaching analysis.
```json
{
  "dialogue": [
    {"speaker": "USER", "transcript": "Let's review the quarterly numbers.", "start_time": 0.0, "end_time": 2.5}
  ],
  "power_axis": "UPWARD"
}
```

---

## 5. CLI & Quick Execution Matrix

| Goal | Command | Description |
| :--- | :--- | :--- |
| **Interactive Live Coaching** | `./record.sh` | Records mic input, performs speech recognition & outputs coaching scorecard. |
| **Custom Silence Timeout** | `./record.sh --silence 2.0` | Allows 2.0s of silence before auto-stopping. |
| **Specific Persona Axis** | `./record.sh --axis UPWARD` | Evaluates communication with C-Suite rubric. |
| **Ambient Conversation Sentinel** | `./nudge.sh` | Low-power background listener that detects meeting onset. |
| **Run All 55+ Unit Tests** | `./venv/bin/pytest -v` | Comprehensive end-to-end regression validation. |
