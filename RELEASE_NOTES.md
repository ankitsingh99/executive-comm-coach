# Release v0.4.0 — Executive Communication Coach

We are pleased to announce the **v0.4.0** release of **Executive Communication Coach**, introducing a secure `WebViewAssetLoader` audio bridge, zero-latency microphone streaming, seamless Android runtime permission bridging, and custom in-app permission dialogs.

---

## 🌟 Key Features & Updates in v0.4.0

### 1. Secure `WebViewAssetLoader` & Web Audio Pipeline
- **Secure Context Origin**: Local assets are served over `https://appassets.androidplatform.net/assets/` via `androidx.webkit:webkit:1.12.1`.
- **Chromium MediaDevices Unlock**: Unlocks `navigator.mediaDevices.getUserMedia()`, `AudioContext`, and `SpeechRecognition` without insecure `file://` scheme restrictions.

### 2. Native Hardware Microphone Permissions Bridge
- **Automatic Resource Delegation**: `WebChromeClient.onPermissionRequest` automatically grants `RESOURCE_AUDIO_CAPTURE` resources to the cockpit when Android OS permissions are active.
- **Bi-directional Bridge API**: `CoachBridgeInterface.hasMicPermission()` and `requestMicPermission()` allow seamless permission requests from the web interface.

### 3. Resilient Audio Engine & Themed Diagnostics
- **`getMicrophoneMediaStream` Fallback**: Intelligent negotiation across modern Web Audio and legacy audio interfaces.
- **Custom Executive Modals**: In-app custom themed dialogs replace native browser `alert()` popups.

---

## Release v0.3.0 — Executive Communication Coach

---

## 🌟 Key Features & Updates in v0.3.0

### 1. Dual Executive Themes (Dark & Light)
- **Executive Dark Mode**: Obsidian/Navy theme (`#070A13` / `#0E1424`), vibrant Indigo/Cyan glow accents, and frosted glass cards.
- **Executive Light Mode**: Clean daylight pearl canvas (`#F0F4F9` / `#FFFFFF`), high-contrast slate typography (`#0F172A`), and refined daylight glass borders.

### 2. Live System Appearance Synchronization
- **OS Theme Listener**: Follows system `prefers-color-scheme` preferences and switches seamlessly in real time.
- **Header & HUD Controls**: Instant segmented toggles in the desktop top bar, mobile topbar pill toggle, and modal visual preview selector.

### 3. Native Android Status & Nav Bar Contrast
- **Adaptive System Insets**: Status and navigation bar icons adapt light/dark contrast automatically on Android 10+ through Android 17+.
- **Jetpack Compose Theming**: Full support for `ThemeMode.DARK`, `ThemeMode.LIGHT`, and `ThemeMode.SYSTEM`.

---

## Release v0.2.0 — Executive Communication Coach

We are pleased to announce the **v0.2.0** release of **Executive Communication Coach**, introducing the native Android companion app, real-time microphone Web Audio waveform visualization, edge-to-edge mobile UI optimizations, and an overhauled executive cockpit.

---

## 🌟 Key Features & Updates in v0.2.0


### 1. Native Android Companion App
- **High-Performance Android APK**: Compiled and optimized for Android 10+ through Android 17+ (`minSdk = 29`, `targetSdk = 35`), running natively on modern flagships like Google Pixel 11.
- **Hardware-Accelerated Web Audio Bridge**: Full Web Audio API `AnalyserNode` integration with `WebChromeClient` audio capture permission delegation.
- **Edge-to-Edge Display & Cutout Handling**: Dynamic safe-area padding for punch-hole cameras and gesture navigation pills (`viewport-fit=cover`).
- **Offline Asset Embedding**: Full coaching studio and scenario engines packaged directly into `assets/index.html`.

### 2. Real-Time Microphone Oscilloscope & Pitch Tracker
- **Acoustic Waveform Canvas**: Replaced synthetic sine waves with real-time time-domain audio data (`getByteTimeDomainData`).
- **Dynamic Energy Gain Boost**: Soft conversational speech is dynamically boosted for crystal-clear visualization.
- **Fundamental Pitch Tracker**: Real-time vocal pitch estimation ($85\text{ Hz} - 450\text{ Hz}$) with active speaker tone feedback.

### 3. Streamlined Executive UX
- **Segmented Workbench**: 3 focused tabs (`Scenarios`, `Live Mic`, `Ambient Gate`) reducing cognitive fatigue.
- **Unified Mobile Bottom Bar**: Instant switching between `Coach`, `Actions`, `Dialogue`, and `Vault`.
- **Categorized Detailed Diagnostics**: Multidimensional tabs for `Overview`, `Dynamics & Arc`, and `Strengths & Fixes`.

---

## 📦 Installation & Deployment

### Run on Android
```bash
# Compile debug APK
cd android && ./gradlew assembleDebug

# Install on connected Google Pixel via ADB
adb install -r android/app/build/outputs/apk/debug/app-debug.apk
```

### Run Python Server & Web App
```bash
# Start backend server
comm-coach-server 8080

# Open in browser
open http://localhost:8080
```

---

## 📊 Verification & Tests
- **55/55 Automated Unit & Integration Tests Passing**
- **Clean Android Gradle Build**: All tasks executed with zero linter errors.
