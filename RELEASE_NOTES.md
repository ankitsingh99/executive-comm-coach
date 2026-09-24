# Release v0.1.0 — Executive Communication Coach

We are pleased to announce the inaugural **v0.1.0** release of **Executive Communication Coach** — an intelligent, privacy-first, on-device AI communication coaching platform for English, Hindi, and code-mixed Hinglish conversations.

---

## 🌟 Key Features & Capabilities

### 1. Acoustic & Multimodal Sensing
- **Native Real-Time Microphone Streaming**: CoreAudio/ALSA streaming via `sounddevice` with instantaneous terminal VU meter and adaptive energy & WebRTC VAD gating.
- **Overlapping Speech & Cross-Talk Detection**: Tracks simultaneous speech, speaker collisions, and turn interruption duration.
- **Acoustic Speaker Biometrics**: Local MFCC feature extraction and cosine-similarity voiceprint memory for persistent speaker identification.
- **Pluggable Multi-ASR Engine Support**:
  - Gemini 2.5 Flash Audio API (cloud)
  - Local Faster-Whisper & FastConformer (100% offline)
  - Sarvam AI Indic Speech Client

### 2. Linguistic Coaching & Executive Persona Ontologies
- **Dynamic Communication Scorecard**:
  - Presence & Delivery Score (15–98)
  - Assertiveness Index (15–98)
  - Active Listening & Interruption Tracking Score (15–98)
  - Filler Word & Hedging Detection (English + Hinglish phonetics)
- **Multi-Register Coaching Rubrics**: 6 communication contexts (*Executive / Upward*, *Peer / Lateral*, *Mentorship / Downward*, *Solo Presentation*, *Casual / Informal*, *Conflict & Negotiation*).
- **Automated Action Item Extraction & Temporal Resolution**: Extracts commitments and automatically resolves relative dates (e.g., *"kal sham 4 baje"* $\rightarrow$ concrete ISO timestamps).

### 3. Privacy & DPDP Act Compliance
- **Zero-Cloud PII Scrubbing**: Pattern-based entity redactor for Aadhaar, PAN, emails, phone numbers, and financial tokens.
- **Statutory DPDP Safeguards**: Dual-tone audio consent chime and Right-to-Erasure voiceprint purge commands.

---

## 📦 Installation & Quickstart

```bash
# Direct pip install from Git
pip install git+https://github.com/ankitsingh99/executive-comm-coach.git@v0.1.0

# Run live mic coaching
executive-comm-coach --axis UPWARD

# Launch Web & Mobile App Emulator
comm-coach-server 8080
```

---

## 📊 Verification & Tests
- **55/55 Automated Unit & Integration Tests Passing** covering acoustic gating, STT engines, metrics scoring, Indic normalizer, and DPDP compliance.
