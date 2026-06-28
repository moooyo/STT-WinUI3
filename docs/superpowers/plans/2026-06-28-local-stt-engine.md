# Local STT Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete, pluggable, two-pass local streaming speech-to-text engine for Windows (WinUI3 / .NET 8) running on bare ONNX Runtime / Windows ML, cross-hardware (CPU/GPU/NPU), multilingual (zh-en code-switching first).

**Architecture:** Layered (spec §4, D5): UI-agnostic `Stt.Core` engine library + `Stt.App` WinUI3. Dependencies flow one direction `App → Core/Audio → Abstractions`. `Stt.Core` never references `Microsoft.UI.*`, so it is headless-testable: `FileAudioCapture` feeds a WAV through the full recognition chain in CI with no mic and no UI. Three threads (audio callback / inference worker / UI) are decoupled by `System.Threading.Channels`. Two-pass mode-2: streaming first pass emits live partials; on VAD endpoint, an offline model re-decodes the whole segment to produce the final.

**Tech Stack:** .NET 8 (built with .NET 10 SDK), C#, ONNX Runtime (`Microsoft.ML.OnnxRuntime` for headless Core/tests; `Microsoft.WindowsAppSDK.ML` swap-in for the app — see Global Constraints), `System.Threading.Channels`, NAudio (WASAPI capture), WinUI3 / Windows App SDK, `CommunityToolkit.Mvvm`, `Microsoft.Extensions.Hosting` (DI), xUnit. Native: `kaldi-native-fbank.dll` via P/Invoke, Silero VAD ONNX, SenseVoice-Small ONNX, streaming Zipformer transducer ONNX.

## Global Constraints

- **Target frameworks:** `Stt.Abstractions`, `Stt.Core`, both test projects = `net8.0`. `Stt.Audio.Windows` = `net8.0-windows`. `Stt.App` = `net8.0-windows10.0.19041.0`. (Spec §4.)
- **Architectures:** x64 + ARM64 only. NPU/optimized-EP code paths are runtime-gated to Windows build 26100 (24H2); below that, fall back to DirectML/CPU. (Spec §4, §9.)
- **Layering (hard rule):** `Stt.Core` MUST NOT reference `Microsoft.UI.*` or any WinUI/Windows App SDK assembly. Only `Stt.App` references Windows App SDK. UI marshalling crosses the boundary only through `IUiDispatcher`. (Spec §4, §6.)
- **Abstractions zero-third-party rule:** `Stt.Abstractions` has zero third-party package references. **Deviation from spec §5.1:** `IExecutionProviderSelector.BuildSessionOptions` returns ORT's `SessionOptions`, an ORT type — so `IExecutionProviderSelector` lives in `Stt.Core` (which references ORT), not in `Stt.Abstractions`. All other interfaces/DTOs/enums stay in Abstractions (they use only spans, primitives, and our own types).
- **ORT package:** Core and tests reference `Microsoft.ML.OnnxRuntime` so they restore/build/test cross-platform in CI. The app composition root may substitute `Microsoft.WindowsAppSDK.ML`; **never reference both** in the same output (double-loads `onnxruntime.dll` — spec §9). Keep all ORT usage behind `SessionOptionsBuilder`/`IExecutionProviderSelector` so the provider swap is one project's concern.
- **Everything fixed-shape (iron rule, spec §9):** DirectML dynamic axes ~5× slower. Streaming encoder = fixed chunk + fixed cache; offline = fixed window or length-bucketed padding.
- **Fail loud (spec §10.2, §14):** On unknown/mismatched feature family or missing required parameter, reject at load time listing what was checked. Never silently default to fbank-80.
- **Audio thread never allocates, never calls ORT. UI thread never blocks, never infers.** Backpressure #1 (audio→worker) uses `TryWrite` + `DropOldest` (never Wait). Backpressure #2 (segment→2nd pass) uses `WriteAsync` + Wait (finals are precious). (Spec §6.)
- **Packaging:** Default unpackaged self-contained (D9 default). MSIX is a later variant. Not built in this plan beyond project structure readiness.
- **Models are user-supplied (D7):** No models committed. `.gitignore` already excludes `*.onnx`, `*.npz`, `models/`, native `*.dll`. Runtime-dependent tests skip when binaries/models are absent.
- **Commit cadence:** Commit after each task's tests pass. Branch off `main` (do not commit straight to `main` if the harness disallows; create `feat/stt-engine`).

---

## Phase 0 — Offline single-pass + infrastructure

Delivers: mic → VAD → kaldi-fbank → SenseVoice (ORT) → text. A usable "speak a sentence, get zh-en mixed text" product. Plus the headless test harness, EP selection/compile-cache/fallback skeleton, and model load validation.

### Task 0.1: Solution + project skeleton + toolchain validation

**Files:**
- Create: `Stt.sln`
- Create: `src/Stt.Abstractions/Stt.Abstractions.csproj`
- Create: `src/Stt.Core/Stt.Core.csproj`
- Create: `src/Stt.Audio.Windows/Stt.Audio.Windows.csproj`
- Create: `src/Stt.App/Stt.App.csproj` (added later in Task 0.12; placeholder ref ok)
- Create: `tests/Stt.Core.Tests/Stt.Core.Tests.csproj`
- Create: `tests/Stt.Pipeline.Tests/Stt.Pipeline.Tests.csproj`
- Create: `Directory.Build.props` (shared LangVersion, Nullable enable, x64;ARM64 platforms)
- Create: `NuGet.config` (nuget.org source)

**Interfaces:**
- Produces: the empty projects + project references (`Core → Abstractions`, `Audio.Windows → Abstractions,Core`, `Core.Tests → Core`, `Pipeline.Tests → Core`).

- [ ] **Step 1:** Create `Directory.Build.props` with `<LangVersion>latest`, `<Nullable>enable`, `<ImplicitUsings>enable`, `<TreatWarningsAsErrors>false`.
- [ ] **Step 2:** Create `src/Stt.Abstractions/Stt.Abstractions.csproj` targeting `net8.0`, no package refs.
- [ ] **Step 3:** Create `src/Stt.Core/Stt.Core.csproj` targeting `net8.0`, PackageRef `Microsoft.ML.OnnxRuntime`, `System.Threading.Channels`, `System.Text.Json`; ProjectRef Abstractions.
- [ ] **Step 4:** Create `tests/Stt.Core.Tests` (net8.0) referencing xUnit + `Microsoft.NET.Test.Sdk` + ProjectRef Core.
- [ ] **Step 5:** Create `tests/Stt.Pipeline.Tests` (net8.0) referencing xUnit + ProjectRef Core.
- [ ] **Step 6:** Create `src/Stt.Audio.Windows` (`net8.0-windows`) PackageRef `NAudio`; ProjectRef Abstractions, Core.
- [ ] **Step 7:** `dotnet new sln` then `dotnet sln add` all projects (App added in 0.12).
- [ ] **Step 8:** Run `dotnet build Stt.Core.Tests` — **Expected: PASS (0 errors)**. This validates net8.0 builds on the .NET 10 SDK and ORT restores. If net8.0 ref packs are missing, run `dotnet restore` first; if still failing, document and escalate.
- [ ] **Step 9:** Add a trivial `UnitTest1` asserting `true` and run `dotnet test tests/Stt.Core.Tests` — Expected: PASS.
- [ ] **Step 10:** Commit `chore: solution skeleton + toolchain validation`.

### Task 0.2: Abstractions — DTOs and enums

**Files:**
- Create: `src/Stt.Abstractions/Audio/AudioFrame.cs`, `SpeechSegment.cs`
- Create: `src/Stt.Abstractions/Decoders/AsrResult.cs`, `DecoderType.cs`, `DecoderCapabilities.cs`
- Create: `src/Stt.Abstractions/Features/AsrFeatureFamily.cs`
- Create: `src/Stt.Abstractions/Pipeline/PartialResult.cs`, `FinalResult.cs`, `PipelineMode.cs`, `PipelineConfig.cs`
- Create: `src/Stt.Abstractions/Ep/EpKind.cs`, `EpPreference.cs`
- Test: `tests/Stt.Core.Tests/Abstractions/DtoTests.cs`

**Interfaces:**
- Produces (exact, verbatim from spec §5.2):
  ```csharp
  public sealed record AudioFrame(float[] Samples, int Count);
  public sealed record SpeechSegment(long StartSample, float[] Samples);
  public sealed record AsrResult(string Text, IReadOnlyList<int> Tokens, IReadOnlyList<float> Timestamps, bool IsFinal);
  public sealed record PartialResult(int SegmentId, string Text);
  public sealed record FinalResult(int SegmentId, string Text);
  public enum AsrFeatureFamily { Auto, KaldiFbankPovey, KaldiFbankLfrCmvn, WhisperLogMel, NemoMel, Mfcc, RawAudioSamples }
  public enum DecoderType { Transducer, Ctc, Nar, Ar }
  public enum PipelineMode { OnePassStreaming, OnePassOffline, TwoPass }
  public enum EpKind { Cpu, DirectML, Cuda, Qnn, OpenVINO, VitisAI }
  [Flags] public enum DecoderCapabilities { None=0, Streaming=1, Offline=2, PartialResults=4, Endpointing=8, Timestamps=16, Multilingual=32 }
  ```
  Plus `public sealed record EpPreference(EpKind Kind, bool AllowFallbackToCpu = true);` and `PipelineConfig` (mode + pass1/pass2 model ids + EP pref + endpoint thresholds).

- [ ] **Step 1:** Write test asserting record equality + flags combination (`(Streaming|Offline).HasFlag(Streaming)`).
- [ ] **Step 2:** Run test — Expected: FAIL (types missing).
- [ ] **Step 3:** Create all DTO/enum files verbatim above.
- [ ] **Step 4:** Run test — Expected: PASS.
- [ ] **Step 5:** Commit `feat(abstractions): DTOs and enums`.

### Task 0.3: Abstractions — service interfaces

**Files:**
- Create: `src/Stt.Abstractions/Audio/IAudioCapture.cs`
- Create: `src/Stt.Abstractions/Features/IFeatureFrontend.cs`
- Create: `src/Stt.Abstractions/Vad/IVad.cs`
- Create: `src/Stt.Abstractions/Decoders/IAsrDecoder.cs`
- Create: `src/Stt.Abstractions/Models/IModelRegistry.cs`, `ModelManifest.cs`
- Create: `src/Stt.Abstractions/Pipeline/ISttPipeline.cs`
- Create: `src/Stt.Abstractions/Common/IUiDispatcher.cs`

**Interfaces:**
- Produces (verbatim spec §5.1, except `IExecutionProviderSelector` which moves to Core per Global Constraints):
  `IAudioCapture`, `IFeatureFrontend`, `IVad`, `IAsrDecoder`, `IModelRegistry`, `ISttPipeline`, `IUiDispatcher`. `ModelManifest` is a class/record matching the §5.3 JSON shape (see Task 0.10).

- [ ] **Step 1:** Write a compile-only test: a `FakeUiDispatcher : IUiDispatcher` and a no-op `IAsrDecoder` stub in the test project, assert they satisfy the interfaces.
- [ ] **Step 2:** Run — Expected: FAIL (interfaces missing).
- [ ] **Step 3:** Create interface files verbatim from §5.1.
- [ ] **Step 4:** Run — Expected: PASS.
- [ ] **Step 5:** Commit `feat(abstractions): service interfaces`.

### Task 0.4: WAV I/O + RingBuffer + Resampler + FrameChunker

**Files:**
- Create: `src/Stt.Core/Audio/WavIo.cs` (read/write 16-bit & float PCM WAV)
- Create: `src/Stt.Core/Audio/RingBuffer.cs` (float ring, single-producer/consumer)
- Create: `src/Stt.Core/Audio/Resampler.cs` (any-rate → 16k mono, linear + anti-alias decimation)
- Create: `src/Stt.Core/Audio/FrameChunker.cs` (accumulate samples, emit fixed-size windows w/ hop)
- Test: `tests/Stt.Core.Tests/Audio/WavIoTests.cs`, `RingBufferTests.cs`, `ResamplerTests.cs`, `FrameChunkerTests.cs`

**Interfaces:**
- Produces:
  - `WavIo.ReadPcm16(string path) -> (float[] samples, int sampleRate, int channels)`; `WavIo.WritePcm16(string path, ReadOnlySpan<float> mono16k, int sampleRate)`.
  - `RingBuffer(int capacity)`: `int Write(ReadOnlySpan<float>)`, `int Read(Span<float>)`, `int Count`.
  - `Resampler.ToMono16k(ReadOnlySpan<float> interleaved, int srcRate, int srcChannels) -> float[]`.
  - `FrameChunker(int windowSize, int hopSize)`: `void Push(ReadOnlySpan<float>)`, `bool TryGetWindow(out float[] window)`.

- [ ] **Step 1:** Write `RingBufferTests`: write 10, read 6, write 8 (wrap), read 12 → FIFO order preserved; overflow truncates and returns written count.
- [ ] **Step 2:** Run — FAIL.
- [ ] **Step 3:** Implement `RingBuffer`.
- [ ] **Step 4:** Run — PASS.
- [ ] **Step 5:** Write `ResamplerTests`: 48k sine → 16k length ≈ src/3; round-trip a 16k mono passthrough is identity; mono-downmix of stereo averages channels.
- [ ] **Step 6:** Implement `Resampler` (downmix to mono, then rational resample; for 48k→16k decimate by 3 with a simple FIR low-pass; general rates via linear interp).
- [ ] **Step 7:** Run — PASS.
- [ ] **Step 8:** Write `FrameChunkerTests`: push 1200 samples with window=512 hop=512 → yields 2 windows, 176 buffered; window=400 hop=160 overlap correctness on a ramp signal.
- [ ] **Step 9:** Implement `FrameChunker`.
- [ ] **Step 10:** Run — PASS.
- [ ] **Step 11:** Write `WavIoTests`: write known float ramp as pcm16 then read back, assert within 1/32768 quantization; assert sampleRate/channels round-trip.
- [ ] **Step 12:** Implement `WavIo`.
- [ ] **Step 13:** Run — PASS. Commit `feat(core): audio primitives (wav, ring, resampler, chunker)`.

### Task 0.5: FileAudioCapture (headless source)

**Files:**
- Create: `src/Stt.Core/Audio/FileAudioCapture.cs` (implements `IAudioCapture`)
- Test: `tests/Stt.Core.Tests/Audio/FileAudioCaptureTests.cs`

**Interfaces:**
- Consumes: `WavIo`, `Resampler`, `IAudioCapture`, `AudioFrame`.
- Produces: `FileAudioCapture(string wavPath, int frameSamples = 512, bool realTime = false)`. Raises `FrameAvailable` with `AudioFrame` (16k mono) in `frameSamples` chunks until EOF, then completes `StartAsync`.

- [ ] **Step 1:** Write test: generate a 1s 16k mono WAV (via `WavIo`), capture with frameSamples=512, collect frames, assert total samples == file length and `realTime:false` returns promptly.
- [ ] **Step 2:** Run — FAIL.
- [ ] **Step 3:** Implement (read file → resample to 16k mono → emit frames from `ArrayPool<float>.Shared`; honor `CancellationToken`).
- [ ] **Step 4:** Run — PASS. Commit `feat(core): FileAudioCapture headless source`.

### Task 0.6: Feature post-processors — LFR, CMVN, PerFeatureNorm

**Files:**
- Create: `src/Stt.Core/Features/Lfr.cs`, `Cmvn.cs`, `PerFeatureNorm.cs`
- Test: `tests/Stt.Core.Tests/Features/LfrTests.cs`, `CmvnTests.cs`, `PerFeatureNormTests.cs`

**Interfaces:**
- Produces (spec §7.2):
  - `Lfr.Apply(ReadOnlySpan<float> feats, int numFrames, int featDim, int lfrM, int lfrN, out int outFrames) -> float[]` — stack `lfrM` frames stepping `lfrN`; left-pad first frame replicate per FunASR; out dim = `featDim*lfrM`.
  - `Cmvn.Apply(Span<float> feats, int numFrames, int dim, ReadOnlySpan<float> negMean, ReadOnlySpan<float> invStddev)` — `x = (x + negMean) * invStddev` in place.
  - `PerFeatureNorm.Apply(Span<float> feats, int numFrames, int dim)` — per-feature mean/std over time (NeMo).

- [ ] **Step 1:** `LfrTests`: 5 frames × 2 dims, lfrM=3 lfrN=2 → expected stacked rows hand-computed; assert outDim=6 and outFrames=3.
- [ ] **Step 2:** Run — FAIL. Implement `Lfr`. Run — PASS.
- [ ] **Step 3:** `CmvnTests`: feats `[1,2,3,4]` dim=2, negMean `[-1,-1]`, invStddev `[0.5,0.5]` → `[0,0.5,1,1.5]`.
- [ ] **Step 4:** Run — FAIL. Implement `Cmvn`. Run — PASS.
- [ ] **Step 5:** `PerFeatureNormTests`: 3 frames × 1 dim ramp → zero mean, unit std (within eps).
- [ ] **Step 6:** Run — FAIL. Implement `PerFeatureNorm`. Run — PASS.
- [ ] **Step 7:** Commit `feat(core): LFR + CMVN + per-feature norm post-processors`.

### Task 0.7: kaldi-native-fbank P/Invoke + KaldiFbankFrontend

**Files:**
- Create: `src/Stt.Core/Features/KaldiNativeFbankInterop.cs` (P/Invoke)
- Create: `src/Stt.Core/Features/KaldiFbankFrontend.cs` (implements `IFeatureFrontend`)
- Create: `src/Stt.Core/Features/FbankOptions.cs`
- Test: `tests/Stt.Core.Tests/Features/KaldiFbankFrontendTests.cs` (skips when native lib absent)
- Doc: `docs/native/kaldi-native-fbank.md` (how to obtain/build the DLL, expected ABI)

**Interfaces:**
- Consumes: `IFeatureFrontend`, `Lfr`, `Cmvn`, `AsrFeatureFamily`.
- Produces: `KaldiFbankFrontend(FbankOptions opts)` with `FeatureDim`, `Extract(ReadOnlySpan<float> pcm, out int numFrames) -> float[]` row-major `[T, dim]`. `FbankOptions`: numBins, sampleRate, dither, snipEdges, isLibrosa(window: povey/hamming), lowFreq, highFreq, normalizeSamples, lfrM/lfrN (optional), cmvn (optional negMean/invStddev). Family A: bins=80, dither=0, snipEdges=false. Family B: + LFR(7,6) + CMVN.

- [ ] **Step 1:** Document the kaldi-native-fbank C ABI to bind (`OnlineFbank`/`KnfOnlineFbank` create/accept-waveform/num-frames-ready/get-frame/free). Provide a thin C shim header in the doc if the upstream lib lacks a stable C export. Note Apache-2.0, win-x64/arm64.
- [ ] **Step 2:** Write `KaldiNativeFbankInterop` `[DllImport("kaldi-native-fbank")]` declarations matching the documented ABI.
- [ ] **Step 3:** Implement `KaldiFbankFrontend.Extract` calling interop; then optional `Lfr`/`Cmvn`.
- [ ] **Step 4:** Write test that is `[SkippableFact]` (Xunit.SkippableFact) — skips if `kaldi-native-fbank.dll` not loadable; otherwise extracts fbank of a 1s tone and asserts shape `[T,80]`, finite, natural-log range ≈ [-20, 20].
- [ ] **Step 5:** Run — PASS or SKIP (documented). Commit `feat(core): kaldi-fbank P/Invoke frontend (A/B families)`.

> Golden numeric validation vs lhotse/funasr (spec §7.3, < 1e-3) is deferred to Task 0.16 (requires Python reference + checked-in golden vectors).

### Task 0.8: SileroVad

**Files:**
- Create: `src/Stt.Core/Vad/SileroVad.cs` (implements `IVad`)
- Create: `src/Stt.Core/Vad/VadOptions.cs`
- Test: `tests/Stt.Core.Tests/Vad/SileroVadTests.cs` (skips when model absent)

**Interfaces:**
- Consumes: `IVad`, `SpeechSegment`, ORT `InferenceSession`.
- Produces: `SileroVad(string modelPath, VadOptions opts)`: `AcceptWaveform(ReadOnlySpan<float> window512)`, `TryDequeueSegment(out SpeechSegment)`, `Reset()`. Maintains hidden state (h/c), threshold + min-silence + min-speech + speech-pad, accumulates speech samples, emits a `SpeechSegment` on trailing silence.

- [ ] **Step 1:** Write `[SkippableFact]` test: with `silero_vad.onnx` present, feed 0.5s silence then 0.5s tone then 1s silence (in 512 windows) → exactly 1 segment whose length ≈ tone duration ± pad. Skip if model absent.
- [ ] **Step 2:** Implement VAD state machine (greedy: prob>thr → in-speech; trailing silence ≥ minSilence → flush segment).
- [ ] **Step 3:** Run — PASS or SKIP. Commit `feat(core): Silero VAD`.

### Task 0.9: Tokenizer (tokens.txt + SentencePiece detokenize)

**Files:**
- Create: `src/Stt.Core/Text/TokenTable.cs` (load tokens.txt: id↔piece)
- Create: `src/Stt.Core/Text/SentencePieceDetokenizer.cs` (pieces → text; `▁`→space, CJK no-space)
- Create: `src/Stt.Core/Text/SpecialTagStripper.cs` (strip `<|...|>` SenseVoice tags)
- Test: `tests/Stt.Core.Tests/Text/TokenTableTests.cs`, `SentencePieceDetokenizerTests.cs`, `SpecialTagStripperTests.cs`

**Interfaces:**
- Produces:
  - `TokenTable.Load(string tokensTxt) -> TokenTable`; `int Count`; `string Piece(int id)`; `bool TryId(string piece, out int id)`.
  - `SentencePieceDetokenizer.Decode(IEnumerable<string> pieces) -> string` (handles `▁` boundary, merges CJK, trims).
  - `SpecialTagStripper.Strip(string text) -> (string clean, IReadOnlyList<string> tags)`.

- [ ] **Step 1:** `TokenTableTests`: parse 3-line tokens file, assert Count and bidirectional lookup.
- [ ] **Step 2:** FAIL → implement → PASS.
- [ ] **Step 3:** `SentencePieceDetokenizerTests`: `["▁hello","▁world"]`→`"hello world"`; `["中","文","▁mix","ed"]`→`"中文 mixed"`.
- [ ] **Step 4:** FAIL → implement → PASS.
- [ ] **Step 5:** `SpecialTagStripperTests`: `"<|zh|><|NEUTRAL|>你好<|woitn|>"` → clean `"你好"`, tags include `zh`.
- [ ] **Step 6:** FAIL → implement → PASS. Commit `feat(core): tokenizer + tag stripper`.

### Task 0.10: ModelManifest + ModelMetadataReader + FeatureFamilyDetector

**Files:**
- Create: `src/Stt.Abstractions/Models/ModelManifest.cs` (if not in 0.3) + `ModelFiles.cs`, `FeatureSpec.cs`, `CapabilityFlags.cs`, `DecodingSpec.cs`
- Create: `src/Stt.Core/Models/ModelMetadataReader.cs` (read ORT `CustomMetadataMap` + input dims)
- Create: `src/Stt.Core/Features/FeatureFamilyDetector.cs`
- Test: `tests/Stt.Core.Tests/Models/ManifestTests.cs`, `FeatureFamilyDetectorTests.cs`

**Interfaces:**
- Produces:
  - `ModelManifest` matching spec §5.3 JSON (System.Text.Json, camelCase): `Id, DisplayName, Version, Family, Runtime[], DecoderType, Files{encoder,decoder,joiner,tokens,model}, Feature{frontEnd,family,sampleRate,featureDim,lfr,cmvn}, Capabilities{streamingCapable,offlineCapable,needsLfrCmvn,multilingual,emitsTimestamps,needsVad}, Languages[], Decoding{defaultMethod,endpointRules}, ProviderSupport[], License`.
  - `ModelManifest.Load(string jsonPath)` / `Save`.
  - `ModelMetadataReader.Read(IReadOnlyDictionary<string,string> meta, IReadOnlyDictionary<string,long[]> inputDims) -> ModelProbe` (model_type, featureDim arbiter, n_mels, layout).
  - `FeatureFamilyDetector.Detect(ModelProbe probe) -> AsrFeatureFamily` per spec §10.1 (80→fbank/whisper disambig by model_type/n_mels/layout; 128→whisper-large-v3; 560→FunASR; layout `[N,T,C]` fbank vs `[N,mels,3000]` whisper).

- [ ] **Step 1:** `ManifestTests`: serialize the §5.3 example, round-trip, assert fields (`family=="transducer"`, `feature.featureDim==80`).
- [ ] **Step 2:** FAIL → implement records + Load/Save → PASS.
- [ ] **Step 3:** `FeatureFamilyDetectorTests`: probe(model_type="sense_voice_ctc", dim=560) → KaldiFbankLfrCmvn; probe(dim=80, model_type="zipformer2") → KaldiFbankPovey; probe(dim=128, layout `[1,128,3000]`) → WhisperLogMel; probe(dim=80, model_type unknown, layout `[1,80,3000]`) → WhisperLogMel; ambiguous unknown → returns `Auto`/throws "Unknown — refuse to default".
- [ ] **Step 4:** FAIL → implement detector → PASS. Commit `feat(core): manifest + metadata reader + family detector`.

### Task 0.11: EP selection — SessionOptionsBuilder, ExecutionProviderSelector, CompiledModelCache

**Files:**
- Create: `src/Stt.Core/Ep/IExecutionProviderSelector.cs` (moved here per Global Constraints)
- Create: `src/Stt.Core/Ep/SessionOptionsBuilder.cs`
- Create: `src/Stt.Core/Ep/ExecutionProviderSelector.cs`
- Create: `src/Stt.Core/Ep/CompiledModelCache.cs`
- Test: `tests/Stt.Core.Tests/Ep/CompiledModelCacheTests.cs`, `ExecutionProviderSelectorTests.cs`

**Interfaces:**
- Produces:
  - `IExecutionProviderSelector.BuildSessionOptions(EpPreference pref, string modelHash) -> SessionOptions`.
  - `CompiledModelCache(string cacheRoot)`: `string ContextPath(string modelHash, string epName, string epVer, string driver)` → stamped `{hash}_{ep}_{ver}_{driver}_ctx.onnx`; `bool TryGetValid(...)`, `void Invalidate(string path)`.
  - `ExecutionProviderSelector` implements `IExecutionProviderSelector`: enumerate `OrtEnv.GetEpDevices()` fresh per call, filter by `EpName`+`HardwareDevice.Type`, append; try/catch → fall back to CPU; wire EPContext options. (ORT-dependent paths guarded; pure logic — cache pathing, fallback decision, stamping — unit-tested.)

- [ ] **Step 1:** `CompiledModelCacheTests`: `ContextPath` is deterministic + includes all stamp fields; `Invalidate` deletes; `TryGetValid` false when file missing.
- [ ] **Step 2:** FAIL → implement cache → PASS.
- [ ] **Step 3:** `ExecutionProviderSelectorTests`: with a fake device list (inject `Func<IEnumerable<EpDeviceInfo>>`), `EpPreference(DirectML)` with no DML device + AllowFallbackToCpu=true → selects CPU; PREFER paths return CPU fallback. (Extract device-filtering into a pure `EpResolver.Resolve(pref, devices) -> chosen` for testability.)
- [ ] **Step 4:** FAIL → implement resolver + selector → PASS. Commit `feat(core): EP selection + compiled-model cache`.

### Task 0.12: NarDecoder (SenseVoice offline)

**Files:**
- Create: `src/Stt.Core/Decoders/GreedyCtc.cs` (argmax + blank-collapse)
- Create: `src/Stt.Core/Decoders/NarDecoder.cs` (implements `IAsrDecoder`)
- Test: `tests/Stt.Core.Tests/Decoders/GreedyCtcTests.cs`, `NarDecoderTests.cs` (model test skips when absent)

**Interfaces:**
- Consumes: `IAsrDecoder`, `IFeatureFrontend` (Family B), `TokenTable`, `SentencePieceDetokenizer`, `SpecialTagStripper`, ORT session, `Lfr`, `Cmvn`.
- Produces:
  - `GreedyCtc.Decode(ReadOnlySpan<float> logProbs, int T, int V, int blank=0) -> int[]` (argmax per frame, collapse repeats, drop blanks).
  - `NarDecoder` buffers features to `InputFinished()`, single `Run`, then GreedyCtc → strip first 4 query slots → detokenize → strip tags → `AsrResult(IsFinal:true)`. `Capabilities = Offline | Multilingual | Timestamps`.

- [ ] **Step 1:** `GreedyCtcTests`: logits for frames `[a,a,blank,b,b]` → `[a,b]`; all-blank → empty; hand-built 3×4 matrix.
- [ ] **Step 2:** FAIL → implement → PASS.
- [ ] **Step 3:** `NarDecoderTests` unit part: inject a fake `IModelRunner` returning canned logits → assert text path (collapse→detok→strip) yields expected string. (Introduce `interface IModelRunner { float[] Run(float[] feats,int T,int dim, out int outT, out int V); }` so decode logic tests without ORT.)
- [ ] **Step 4:** FAIL → implement NarDecoder + OrtModelRunner (real ORT) → PASS (unit) / SKIP (real-model integration).
- [ ] **Step 5:** Commit `feat(core): SenseVoice NAR decoder + greedy CTC`.

### Task 0.13: ModelRegistry + ModelLoader (5-layer fail-loud validation)

**Files:**
- Create: `src/Stt.Core/Models/ModelRegistry.cs` (implements `IModelRegistry`)
- Create: `src/Stt.Core/Models/ModelLoader.cs`
- Create: `src/Stt.Core/Models/ModelValidation.cs` (the 5 gates)
- Test: `tests/Stt.Core.Tests/Models/ModelRegistryTests.cs`, `ModelValidationTests.cs`

**Interfaces:**
- Consumes: `IModelRegistry`, `ModelManifest`, `FeatureFamilyDetector`, `ModelMetadataReader`, `TokenTable`.
- Produces:
  - `ModelRegistry(string modelsRoot, IEnumerable<string> extraPaths)`: `List/Get/ImportFromFolder/Remove` + `ResolveCombination(pass1Id, pass2Id, PipelineMode) -> ResolvedPipelineModels` (validates capability legality: pass1 must be streamingCapable; 2-pass/offline needs VAD; pass2 must be offlineCapable).
  - `ImportFromFolder`: load `manifest.json` if present; else infer from file naming (encoder/decoder/joiner→transducer; single model.onnx→ctc/paraformer) + ONNX metadata → tentative manifest for user confirmation.
  - `ModelValidation.Validate(ModelManifest, ModelProbe, TokenTable) -> ValidationReport` implementing spec §10.2 gates 1-4 (5 optional self-test deferred): family detect; expectedDim == encoder input dim else reject; Family B requires lfr/cmvn and `80*lfr==featDim`; tokens line count == vocab_size.

- [ ] **Step 1:** `ModelValidationTests`: mismatched dim (manifest says 80, probe says 560) → rejected listing checks; Family B missing cmvn → rejected; tokens count ≠ vocab → rejected; correct combo → passes.
- [ ] **Step 2:** FAIL → implement `ModelValidation` → PASS.
- [ ] **Step 3:** `ModelRegistryTests`: create temp folder fixtures (manifest + fake tokens.txt; and a manifest-less folder with `encoder.onnx`/`decoder.onnx`/`joiner.onnx` empty files) → `ImportFromFolder` infers transducer family; `ResolveCombination` rejects "Whisper as pass1" (offline-only) with a clear reason.
- [ ] **Step 4:** FAIL → implement registry + loader → PASS. Commit `feat(core): model registry + loader + 5-layer validation`.

### Task 0.14: EndpointDetector + offline pipeline (OnePassOffline)

**Files:**
- Create: `src/Stt.Core/Pipeline/EndpointDetector.cs`
- Create: `src/Stt.Core/Pipeline/SttPipeline.cs` (implements `ISttPipeline`; Phase 0 wires OnePassOffline path)
- Create: `src/Stt.Core/Pipeline/PipelineFactory.cs`
- Test: `tests/Stt.Pipeline.Tests/OfflinePipelineTests.cs`, `EndpointDetectorTests.cs`

**Interfaces:**
- Consumes: `IAudioCapture`, `IVad`, `IFeatureFrontend`, `IAsrDecoder`(NAR), `IUiDispatcher`, Channels.
- Produces:
  - `EndpointDetector(float minTrailingSilenceSec, float maxUtteranceSec)`: `bool Update(bool isSpeech, double elapsedSec)` → true on silence-rule or max-duration.
  - `SttPipeline(PipelineConfig, IAudioCapture, IVad, frontend, decoder, IUiDispatcher)`: `StartAsync/StopAsync`, raises `Final` (and `Partial` in Phase 1). OnePassOffline: audio → VAD segments → frontend → NAR decode → `Final`.

- [ ] **Step 1:** `EndpointDetectorTests`: speech then 1.2s silence (minSilence=1.0) → endpoint; continuous speech past maxUtterance → endpoint; short blips don't trigger.
- [ ] **Step 2:** FAIL → implement → PASS.
- [ ] **Step 3:** `OfflinePipelineTests`: wire `FileAudioCapture` + a **fake** `IVad` (emits one segment) + **fake** frontend + **fake** NAR decoder (returns "你好 world") + capturing `IUiDispatcher` → run StartAsync → assert exactly one `Final("你好 world")`. Tests channel orchestration without native deps.
- [ ] **Step 4:** FAIL → implement SttPipeline OnePassOffline + factory → PASS. Commit `feat(core): endpoint detector + offline pipeline`.

### Task 0.15: WASAPI capture (Stt.Audio.Windows)

**Files:**
- Create: `src/Stt.Audio.Windows/WasapiAudioCapture.cs` (implements `IAudioCapture`)
- Test: `tests/Stt.Core.Tests` cannot run mic in CI — provide a manual smoke note in `docs/native/audio-capture.md`.

**Interfaces:**
- Consumes: `IAudioCapture`, NAudio `WasapiCapture`, `Resampler`.
- Produces: `WasapiAudioCapture(int? deviceIndex = null)`: opens default capture, converts to 16k mono float, emits `AudioFrame` from `ArrayPool`. Catches `UnauthorizedAccessException` (mic denied → surfaces event).

- [ ] **Step 1:** Implement using NAudio `WasapiCapture`/`MMDeviceEnumerator`; resample device format → 16k mono.
- [ ] **Step 2:** `dotnet build src/Stt.Audio.Windows` — Expected: PASS (build-only; runtime needs a device).
- [ ] **Step 3:** Commit `feat(audio): WASAPI capture (NAudio)`.

### Task 0.16: Feature golden test harness (deferred numeric validation)

**Files:**
- Create: `tests/Stt.Core.Tests/Features/GoldenFeatureTests.cs` (`[SkippableFact]`, loads checked-in golden `.bin` when present)
- Create: `tests/golden/README.md` (how to regenerate goldens via lhotse/funasr Python)
- Create: `scripts/gen_golden_features.py` (reference generator — not run in CI)

**Interfaces:**
- Consumes: `KaldiFbankFrontend`, `Lfr`, `Cmvn`.
- Produces: golden comparison asserting max-abs < 1e-3, mean-abs < 1e-4 (spec §7.3) when golden vectors + native lib both present; else SKIP with message.

- [ ] **Step 1:** Write the `scripts/gen_golden_features.py` generator (lhotse fbank for Family A; funasr wav_frontend for Family B) writing `input.wav` + `featsA.bin`/`featsB.bin` + shape sidecars.
- [ ] **Step 2:** Write `GoldenFeatureTests` that loads goldens, runs C# frontend, diffs.
- [ ] **Step 3:** Run — SKIP (no goldens/native in CI). Document regeneration. Commit `test(core): feature golden harness (skippable)`.

### Task 0.17: WinUI3 App shell — DI host, MainPage (offline transcription)

**Files:**
- Create: `src/Stt.App/Stt.App.csproj` (`net8.0-windows10.0.19041.0`, WinUI3, `WindowsAppSDKSelfContained`, `Microsoft.WindowsAppSDK`, `CommunityToolkit.Mvvm`, `Microsoft.Extensions.Hosting`)
- Create: `App.xaml`/`App.xaml.cs` (Host builder + DI registration)
- Create: `MainWindow.xaml`/`.cs`
- Create: `Views/MainPage.xaml`/`.cs`
- Create: `ViewModels/MainViewModel.cs` (`[ObservableProperty]` partial/final lists; `[RelayCommand]` Start/Stop)
- Create: `Services/DispatcherQueueUiDispatcher.cs` (implements `IUiDispatcher` wrapping `DispatcherQueue.TryEnqueue`)
- Create: `Services/SttOptions.cs` (`IOptions<SttOptions>`)
- Reference winui skills: **use `winui:winui-dev-workflow` to build/run**, `winui:winui-design` for XAML.

**Interfaces:**
- Consumes: `ISttPipeline`, `IModelRegistry`, `IUiDispatcher`, `IAudioCapture` (WASAPI in app DI).
- Produces: a window with a transcript list (partials grey/italic, finals black), Start/Stop, device/language selectors, "behind" indicator, copy/export. Phase 0 shows finals only.

- [ ] **Step 1:** Use `winui:winui-dev-workflow` skill; scaffold the WinUI3 project (the skill knows the BuildAndRun workflow + prerequisites).
- [ ] **Step 2:** Register DI: `IModelRegistry`, `IExecutionProviderSelector`, `IUiDispatcher`→`DispatcherQueueUiDispatcher`, `IAudioCapture`→`WasapiAudioCapture`, `ISttPipeline`→`SttPipeline`, `MainViewModel`.
- [ ] **Step 3:** `MainViewModel.StartRecording` (`AsyncRelayCommand`) subscribes pipeline `Final`/`Partial` (already on UI thread via `IUiDispatcher`) → updates observable collections.
- [ ] **Step 4:** Build & launch via the skill's workflow — Expected: app opens, Start/Stop toggles, no crash with no model selected (shows "select a model" empty state).
- [ ] **Step 5:** Commit `feat(app): WinUI3 shell + MainPage offline transcription`.

### Task 0.18: Phase 0 integration smoke + verification

**Files:**
- Create: `docs/native/SETUP.md` (how to drop in `kaldi-native-fbank.dll`, `silero_vad.onnx`, a SenseVoice model folder + manifest to run end-to-end)
- Modify: `README.md` (project overview, build, run, model setup)

- [ ] **Step 1:** Run full `dotnet build Stt.sln` (App excluded on non-Windows CI via solution filter or `-p:`) — document which projects build where.
- [ ] **Step 2:** Run `dotnet test` — all unit tests green; native/model tests SKIP with clear messages.
- [ ] **Step 3:** Use `superpowers:verification-before-completion` to confirm Phase 0 claims with command output.
- [ ] **Step 4:** Write `README.md` + `SETUP.md`. Commit `docs: Phase 0 setup + verification`.

---

## Phase 1 — Streaming first pass → two-pass

Delivers: streaming Zipformer transducer greedy decode aligned to sherpa, hot loop with IO-binding + state double-buffering, endpoint-driven two-pass, live partial subtitles UI.

### Task 1.1: EncoderStateFactory (metadata-driven zero init)

**Files:** Create `src/Stt.Core/Decoders/EncoderStateFactory.cs`; Test `tests/Stt.Core.Tests/Decoders/EncoderStateFactoryTests.cs`.

**Interfaces:** Produces `EncoderStateFactory.BuildInitialStates(ZipformerGeometry geo) -> List<OrtValue>` and `ZipformerGeometry.FromMetadata(IReadOnlyDictionary<string,string>)`. Reads `encoder_dims/query_head_dims/value_head_dims/num_heads/num_encoder_layers/cnn_module_kernels/left_context_len` (comma arrays) + `T/decode_chunk_len/context_size/vocab_size`. State count `m*6+2` (m=Σnum_encoder_layers): per layer 6 caches + global embed_states + processed_lens(int64). (Spec §8.2.)

- [ ] Step 1: Test parses a representative metadata dict (real icefall export values) → asserts state tensor count == `m*6+2` and each cache's shape per the §8.2 formulas (hand-derived expected dims).
- [ ] Step 2: FAIL → implement geometry parse + shape derivation → PASS.
- [ ] Step 3: Commit `feat(core): encoder state factory (metadata-driven)`.

### Task 1.2: TransducerDecoder greedy (state mgmt, testable without ORT)

**Files:** Create `src/Stt.Core/Decoders/TransducerDecoder.cs`, `src/Stt.Core/Decoders/TransducerGreedy.cs`; Test `TransducerGreedyTests.cs`.

**Interfaces:** Consumes `EncoderStateFactory`, `IModelRunner`-style seams for encoder/decoder/joiner. Produces `TransducerGreedy.Step(encoderFrame, ref decoderOut, history) -> token?` implementing: per encoder frame → joiner(enc, dec) → argmax → non-blank(≠0) append + rerun predictor; blank reuse; cache decoder_out across chunks. `Capabilities = Streaming|PartialResults|Endpointing|Timestamps|Multilingual`.

- [ ] Step 1: Test the greedy logic with **fake** joiner/predictor delegates (no ORT): scripted argmax sequence `[blank, 5, 5(dup-but-transducer-allows), blank, 7]` → assert emitted tokens `[5,5,7]` and predictor reruns only on non-blank.
- [ ] Step 2: FAIL → implement `TransducerGreedy` pure logic → PASS.
- [ ] Step 3: Implement `TransducerDecoder` wiring real ORT encoder/decoder/joiner sessions + double-buffered states behind the same seam.
- [ ] Step 4: Commit `feat(core): streaming transducer greedy decoder`.

### Task 1.3: Streaming hot loop — IO binding + state double-buffer

**Files:** Create `src/Stt.Core/Decoders/OrtStreamingSession.cs` (IO binding, preallocated `OrtValue`, A/B state swap). Test: `[SkippableFact]` alignment test scaffold.

**Interfaces:** Produces `OrtStreamingSession` managing `OrtIoBinding`, preallocated input/output `OrtValue`s, per-chunk copy of new features only, state ref-swap (read A / write B then swap — ORT can't read+write same buffer). Dispose unpins. (Spec §8.6.)

- [ ] Step 1: Implement IO-binding session wrapper.
- [ ] Step 2: `[SkippableFact]` numeric-alignment test: with a real streaming Zipformer model + a WAV, run chunked and assert 1-best tokens match a checked-in sherpa-onnx reference (spec §8.2 "align to sherpa before trusting"). SKIP when model/reference absent.
- [ ] Step 3: Commit `feat(core): streaming hot loop (IO binding + double buffer)`.

### Task 1.4: Two-pass pipeline + partials

**Files:** Modify `src/Stt.Core/Pipeline/SttPipeline.cs` (add OnePassStreaming + TwoPass paths, second channel, two workers). Test `tests/Stt.Pipeline.Tests/TwoPassPipelineTests.cs`, `BackpressureTests.cs`.

**Interfaces:** Produces: first-pass worker (LongRunning) streaming-decodes → emits `Partial` (merged ~100-150ms) → on endpoint pushes `SpeechSegment` to bounded `Channel` (FullMode=Wait); second-pass worker (LongRunning, low priority) re-decodes → emits `Final` replacing partial by `SegmentId`. Backpressure #1 DropOldest + dropped-count callback; #2 WriteAsync+Wait.

- [ ] Step 1: `BackpressureTests`: flood channel #1 → asserts DropOldest increments dropped counter and never blocks the writer; channel #2 WriteAsync awaits (no drop).
- [ ] Step 2: FAIL → implement worker/channel wiring with **fake** streaming + NAR decoders → PASS.
- [ ] Step 3: `TwoPassPipelineTests`: fake streaming decoder emits partials "你" → "你好"; endpoint → fake NAR returns "你好 world" → assert `Partial` events then one `Final` with matching `SegmentId`.
- [ ] Step 4: FAIL → implement TwoPass path → PASS. Commit `feat(core): two-pass pipeline + partials + backpressure`.

### Task 1.5: App — live partial subtitles + pipeline/settings pages

**Files:** Modify `Views/MainPage.xaml` (grey partial line above finals); Create `Views/ModelManagerPage.xaml`, `Views/SettingsPage.xaml` + VMs. Use `winui:winui-design` + `winui:winui-ui-testing`.

**Interfaces:** Consumes `IModelRegistry` (model manager: list/import/badges/validation results/remove), pipeline config (1/2-pass + per-pass model assignment with illegal combos greyed + hover reason + EP preference + VAD/endpoint thresholds).

- [ ] Step 1: ModelManagerPage: FolderPicker import → `ImportFromFolder` → show capability badges + validation report; remove.
- [ ] Step 2: SettingsPage: mode + per-pass model pickers (illegal greyed via capability flags) + EP pref + thresholds bound to `SttOptions`.
- [ ] Step 3: MainPage partial line bound to streaming partial.
- [ ] Step 4: UI test via `winui:winui-ui-testing`; commit `feat(app): model manager + settings + live partials`.

---

## Phase 2 — GPU (DirectML)

### Task 2.1: DirectML EP wiring + fixed-shape enforcement
**Files:** Modify `ExecutionProviderSelector`/`SessionOptionsBuilder` to append DirectML when selected/available; add `make_dynamic_shape_fixed`/`AddFreeDimensionOverrideByName` handling for fixed dims. Prioritize 2nd-pass (SenseVoice) encoder on DML. Tests: selector chooses DML when device present (fake device list), CPU fallback otherwise.
- [ ] Implement + unit-test selection logic; `[SkippableFact]` perf/parity smoke with a real DML device. Commit.

### Task 2.2: Quantization-variant selection
**Files:** `ModelRegistry` picks int8/fp16 variant by EP capability + manifest `providerSupport`. Test variant resolution logic. Commit.

---

## Phase 3 — NPU (+ optional Whisper-genai plugin)

### Task 3.1: NPU EP gating (QNN/OpenVINO/VitisAI) for static NAR encoder
**Files:** Runtime-gate to build 26100+; `SetEpSelectionPolicy(PREFER_NPU)` path; require fully static + quantized + no Loop/If. `[SkippableFact]` (needs NPU hardware). Document QNN no-dynamic-shapes constraint. Commit.

### Task 3.2: Optional WhisperGenAiDecoder plugin (Ar capability)
**Files:** New optional `src/Stt.Plugins.WhisperGenAi/` referencing `Microsoft.ML.OnnxRuntimeGenAI` — drop-in `IAsrDecoder` with `Ar` capability, not in core path (spec §8.5). Build-only; not wired into default pipeline. Commit.

---

## Self-Review (spec coverage)

- §1 goals → Phases 0-3. §2 D1-D10 → Global Constraints + EP/decoder tasks. §3/§16 phases → plan phases. §4 structure → Task 0.1 + per-component tasks. §5 abstractions → 0.2/0.3. §6 threading → 0.14 + 1.4 (channels/backpressure). §7 features → 0.6/0.7/0.16. §8 decoders → 0.12 (NAR), 1.1-1.3 (transducer), 3.2 (AR). §9 EP → 0.11, Phase 2/3. §10 model mgmt/validation → 0.10/0.13. §11 modes → 0.14/1.4. §12 UI → 0.17/1.5. §13 packaging → Global Constraints + project setup (full MSIX deferred, documented). §14 errors → fail-loud in 0.13, fallback in 0.11, teardown in 1.4. §15 tests → throughout (headless + golden + alignment + validation + pipeline). §17 risks → mitigations embedded (golden, sherpa-align, fixed-shape, stamped cache). §18 open decisions → packaging default unpackaged (documented); language breadth zh-en first; CTC optional (deferred).

**Known deferrals (documented, not silent):** golden numeric vectors + sherpa alignment references require Python tooling + real models → tests are `[SkippableFact]`. Real end-to-end recognition requires user-supplied native DLLs + ONNX models (D7). Full MSIX packaging variant is structure-ready but not built. Whisper CTC streaming alternative (§8.3) deferred pending model availability.
