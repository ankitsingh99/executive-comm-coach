# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] - 2026-09-24

### Added
- **Native CoreAudio/ALSA Streaming**: Real-time microphone capture via `sounddevice` with dynamic VU level meter.
- **Acoustic Gating & WebRTC VAD**: Ambient voice activity detection with $<2.5\%$ CPU overhead.
- **Cross-Talk & Interruption Detection**: Identifies overlapping speech turns and calculates collision duration.
- **Acoustic Voiceprint Memory**: Local 13-dim MFCC feature extraction with cosine-similarity speaker identification.
- **Multi-Register Coaching Rubrics**: 6 executive communication modes (`UPWARD`, `LATERAL`, `DOWNWARD`, `SOLO`, `CASUAL`, `CONFLICT`).
- **Indic & Hinglish Linguistic Engine**: Devanagari transliteration, code-mixed phonetic normalization, and vocal noise filtering.
- **Action Item & Temporal Resolver**: Extracts deliverables and resolves relative deadlines (*"kal sham 4 baje"*, *"by EOD"*) to absolute ISO 8601 timestamps.
- **DPDP Act Compliance & Security**: Zero-cloud PII redaction (Aadhaar, PAN, phone, email) and Right-to-Erasure voiceprint commands.
- **Packaging & CLI Executables**: Direct installation via Git and pip with `executive-comm-coach`, `comm-coach`, and `comm-coach-server` console commands.
- **GitHub Packages & Container Registry (GHCR)**: Automated workflow and Docker image support.
- **Comprehensive Automated Test Suite**: 55 unit and integration tests passing across all engine components.
