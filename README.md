# Local Streaming Speech-to-Text (STT) Engine

A local, offline, privacy-preserving streaming speech-to-text engine for Windows — built on bare
ONNX Runtime / Windows ML, cross-hardware (CPU / GPU / NPU), multilingual with a focus on
**Chinese–English code-switching**. Two-pass design: a streaming first pass emits live partial
subtitles, and on end-of-sentence an offline model re-decodes the whole segment for the final
text.

> The repository is named `tts-winui3` for historical reasons; the actual target is **STT**
> (speech-to-text), not TTS. See [the design spec](docs/superpowers/specs/2026-06-28-local-stt-engine-design.md).

## Status

**Phases 0–3 are implemented and both recognition paths are verified against sherpa-onnx.** The
**offline** path (SenseVoice-Small + Silero VAD, locally built kaldi-fbank shim) produces correct
zh/en transcription (e.g. `zh.wav → 开饭时间早上九点至下午五点`, `en.wav → the tribal chieftain
called for the boy and presented him with fifty pieces of …`). The **streaming** path (Zipformer2
transducer) produces a transcript that is an **exact match** with the sherpa-onnx reference
(the spec §8.2 "align to sherpa before trusting" gate). GPU/NPU EP foundations and the optional
Whisper plugin round out Phases 2–3. To run end-to-end you supply the native fbank shim + models
(spec D7) — see [SETUP](docs/native/SETUP.md). Headless tests (no native/models): **116 passing,
8 skipped**; with the shim + Silero VAD + SenseVoice + a streaming Zipformer2 + the golden vectors
present: **131 passing, 0 skipped** (Core 124 + Pipeline 7) — every integration test (fbank vs
lhotse, Whisper mel vs openai-whisper, VAD, offline transcription, streaming sherpa-alignment)
runs and passes.

**Feature families (spec §7).** Implemented front-ends: **A** kaldi-fbank Povey (icefall
Zipformer/CTC), **B** kaldi-fbank + LFR + CMVN (FunASR Paraformer / SenseVoice), **C** Whisper
log-mel 80/128 (OpenAI Whisper / Qwen audio), **D** NeMo librosa-mel + per-feature norm (NVIDIA
Parakeet / Canary / GigaAM). All four extract via the native knf shim; C/D were verified against
openai-whisper and the per-feature-norm contract respectively.

| Phase | Scope | State |
|---|---|---|
| 0 | mic → VAD → kaldi-fbank → SenseVoice (ORT) → text; infra; load validation; UI | ✅ implemented + verified |
| 1 | streaming Zipformer transducer → two-pass; live partials | ✅ implemented + verified (exact sherpa-onnx alignment) |
| 2 | DirectML GPU — variant selection, fixed-shape, OS gating; app-side EP wiring | ✅ Core foundations + [docs](docs/native/execution-providers.md) |
| 3 | NPU (QNN/OpenVINO/VitisAI) gating; optional Whisper-genai plugin | ✅ gating logic + `Stt.Plugins.WhisperGenAi` |

> The Whisper plugin is the drop-in for the reserved `DecoderType.Ar` family (spec §8.5); its runtime
> `DecoderCapabilities` are `Offline | Multilingual` (there is no `Ar` capability flag). It is excluded
> from `Stt.slnx` and the core path by design — a [test](tests/Stt.Core.Tests/Abstractions/PluginIsolationTests.cs)
> enforces that neither `Stt.Core` nor `Stt.Abstractions` references it or onnxruntime-genai.

The WinUI 3 front end (spec §12) follows Fluent Design conventions: Mica backdrop with a custom
title bar, adaptive `NavigationView`, virtualized transcript list, `InfoBar` for errors/validation,
control `Header` labels, the type ramp for text, and `AutomationProperties` on interactive controls.
It implements live partials + finals, microphone selection, copy/export, model import with
validation feedback, and confirm-before-delete. (Language is auto-detected by SenseVoice, so there is
no language picker by design.)

## Architecture (spec §4, D5)

Layered, one-directional `App → Core/Audio → Abstractions`:

```
Stt.slnx
├─ src/
│  ├─ Stt.Abstractions   (net10.0)               interfaces + DTOs + enums, zero third-party deps
│  ├─ Stt.Core           (net10.0)               engine: Audio, Features, Vad, Decoders, Ep, Models, Pipeline
│  ├─ Stt.Audio.Windows  (net10.0-windows)       WASAPI capture (NAudio)
│  └─ Stt.App            (net10.0-windows10.0.19041.0, WinUI3)  full-trust unpackaged app
└─ tests/
   ├─ Stt.Core.Tests     (net10.0, xUnit)        WAV → text, feature/decoder/validation units
   └─ Stt.Pipeline.Tests (net10.0, xUnit)        channel backpressure, lifecycle, offline pipeline
```

`Stt.Core` never references `Microsoft.UI.*`, so the full recognition chain is headless-testable:
`FileAudioCapture` feeds a WAV through VAD → features → decode with no microphone and no UI.

## Build & test

Requires the **.NET 10 SDK**. The class libraries and tests target `net10.0` (AnyCPU); the app is
`net10.0-windows10.0.19041.0` (x64 / ARM64) on **Windows App SDK 2.2** (self-contained, unpackaged).

```bash
# Headless engine + tests (cross-platform-ish; ONNX native is win/x64 at runtime)
dotnet test tests/Stt.Core.Tests/Stt.Core.Tests.csproj
dotnet test tests/Stt.Pipeline.Tests/Stt.Pipeline.Tests.csproj
```

### Building the WinUI app

The WinUI app needs the MRT/PRI build task that ships with **Visual Studio's MSBuild** (the
`dotnet` SDK alone lacks it). Build with VS MSBuild:

```powershell
& "C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\amd64\MSBuild.exe" `
  src\Stt.App\Stt.App.csproj -p:Platform=x64 -p:Configuration=Debug -restore
```

To **run** the app, install the WinUI tooling + `winapp` CLI via the `/winui-setup` workflow, then
use `winapp run` / `BuildAndRun.ps1` (never launch the packaged exe directly).

### Packaging (opt-in MSIX)

The default build is **unpackaged** (spec §13 / D9). An opt-in MSIX variant exists for Store/enterprise
distribution; it is dormant unless you pass `-p:Packaged=true` (the `Package.appxmanifest` is only
auto-included when packaging is enabled, so the unpackaged path is untouched):

```powershell
& "C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\amd64\MSBuild.exe" `
  src\Stt.App\Stt.App.csproj -p:Platform=x64 -p:Configuration=Release -p:Packaged=true -restore
# → bin\x64\Release\net10.0-windows10.0.19041.0\win-x64\AppPackages\...\Stt.App_1.0.0.0_x64.msix (unsigned)
```

This produces an **unsigned** `.msix`. To install it you must sign it with a trusted certificate —
use the `winui:winui-packaging` skill (`winapp cert generate` → trust → `winapp sign`). Update the
`Publisher` in `Package.appxmanifest` to match the certificate subject first. Store submission needs a
Partner Center identity (external).

## Running end-to-end

Phase 0 needs three user-supplied pieces (none are committed — D7):

1. The native **kaldi-fbank shim** (`kaldi_native_fbank_shim.dll`) — see
   [docs/native/kaldi-native-fbank.md](docs/native/kaldi-native-fbank.md).
2. A **Silero VAD** ONNX model.
3. A **SenseVoice-Small** ONNX model folder (+ `tokens.txt`, CMVN in metadata).

Full instructions: [docs/native/SETUP.md](docs/native/SETUP.md).

## Design & plan

- Design spec: [docs/superpowers/specs/2026-06-28-local-stt-engine-design.md](docs/superpowers/specs/2026-06-28-local-stt-engine-design.md)
- Implementation plan: [docs/superpowers/plans/2026-06-28-local-stt-engine.md](docs/superpowers/plans/2026-06-28-local-stt-engine.md)
