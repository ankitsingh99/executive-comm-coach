# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.3.0] - 2026-09-26

### Added
- **Dark & Light Mode Support**: Dual executive themes tailored for high-contrast presentation:
  - **🌙 Executive Dark**: Deep obsidian canvas (`#070A13`), neon indigo/cyan glow accents, and frosted glass cards.
  - **☀️ Executive Light (Daylight)**: Crisp daylight pearl canvas (`#F0F4F9`), pure white surfaces, and rich slate typography (`#0F172A`).
- **System Theme Auto-Synchronization**:
  - Live OS color scheme detection (`prefers-color-scheme`) with automatic background switching.
  - Desktop header segmented switch (`Dark | Light | System`).
  - Mobile HUD topbar one-tap theme cycle toggle.
  - Settings & Privacy modal visual preview cards for Dark, Light, and System sync.
- **Android System Bar Synchronization**: Dynamic Android Status Bar & Navigation Bar light/dark contrast controller via `WindowCompat.getInsetsController`.
- **Jetpack Compose ThemeMode**: Expanded `ExecCoachTheme` with `ThemeMode` enum (`DARK`, `LIGHT`, `SYSTEM`).

---

## [0.2.0] - 2026-09-26


### Added
- **Native Android Companion App**: Assembled debug APK (`app-debug.apk`) for Android 10+ through Android 17+ (API 35+), tested and optimized on Google Pixel 11.
- **Hardware-Accelerated Web Audio Engine**: Integrated HTML5 Web Audio `AnalyserNode` with `WebChromeClient` audio capture permission delegation on Android.
- **Real-Time Microphone Waveform Oscilloscope**: Live time-domain audio visualizer with dynamic RMS energy gain and fundamental vocal pitch estimation ($85\text{ Hz} - 450\text{ Hz}$).
- **Edge-to-Edge Display & Cutout Support**: Added safe area insets for Android status bars, camera punch holes, and bottom gesture navigation pills.
- **Streamlined Executive UI / UX**:
  - Segmented left workbench with tabs for Scenarios, Live Mic, and Ambient Gating.
  - Unified mobile bottom navigation bar (`Coach`, `Actions`, `Dialogue`, `Vault`).
  - Categorized Detailed Diagnostics modal (`Overview`, `Dynamics & Arc`, `Strengths & Fixes`).
- **Comprehensive Documentation**: Added Android ADB deployment guide, sideloading instructions, and architectural breakdown to `README.md` and `ARCHITECTURE.md`.

### Fixed
- Fixed stray character syntax error in `emulator/index.html`.
- Fixed Android WebView 980px desktop emulation scale bug on high-DPI devices by configuring `useWideViewPort = false` and `viewport-fit=cover`.
- Updated Android build configuration with `useLegacyPackaging = true` for seamless native library linking.

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
