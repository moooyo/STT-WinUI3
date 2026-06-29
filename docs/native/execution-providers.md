# Execution providers & hardware acceleration (Phases 2–3)

Spec §9. The engine selects an ONNX Runtime execution provider per component and degrades
gracefully. Phase 0/1 ship **CPU**; Phases 2–3 add **DirectML (GPU)** and **NPU**.

## Why the EP wiring lives at the app boundary

`Stt.Core` references the base `Microsoft.ML.OnnxRuntime` (CPU) so the engine and tests restore and
run anywhere. DirectML and the NPU EPs come from **Windows ML** (`Microsoft.WindowsAppSDK.ML`).
Per spec §9 you must **not** load two copies of `onnxruntime.dll`, so the app swaps the ORT
provider package and surfaces real devices to `ExecutionProviderSelector` through an injected
device enumerator. The Core logic (resolution, fallback, cache stamping, variant selection,
fixed-shape, OS gating) is provider-agnostic and unit tested.

## Runtime versions & the WinML single-ORT (current state)

`Stt.Core` references **`Microsoft.ML.OnnxRuntime` 1.27.0** (managed bindings). DirectML + vendor NPU EPs come
from **Windows ML** (`Microsoft.Windows.AI.MachineLearning` 2.1.70), which bundles `onnxruntime.dll` + `DirectML.dll`
in `runtimes/<rid>/native/`. The app honors the §9 iron rule (one `onnxruntime.dll`) by making **Windows ML own
the native** and excluding the base package's native:

```xml
<PackageReference Include="Microsoft.Windows.AI.MachineLearning" Version="2.1.70" />
<PackageReference Include="Microsoft.ML.OnnxRuntime" Version="1.27.0" ExcludeAssets="native" />
```

`App.OnLaunched` calls `ExecutionProviderCatalog.GetDefault().EnsureAndRegisterCertifiedAsync()` in the
background; `OrtEpEnumerator` then surfaces DirectML/NPU via `OrtEnv.GetEpDevices()` and `SessionOptionsBuilder`
appends the chosen device. (If WinML's ORT is older than the 1.27 managed bindings, align by pinning the engine
to the WinML ORT version — autoEP APIs are stable since 1.22, and all calls are wrapped with CPU fallback.)

## What is implemented in Core (provider-agnostic, tested)

- `OrtEpEnumerator` — `OrtEnv.GetEpDevices()` → `EpDeviceInfo` (with backing `OrtEpDevice`); defensive → CPU.
- `SessionOptionsBuilder.Build` — appends the resolved EP via `AppendExecutionProvider(env, devices)`, gated by
  `OsCapabilities` (DirectML ≥1809, NPU ≥24H2); pipeline builders retry on CPU if the EP session fails.
- `EpResolver` — preference → device with CPU fallback (spec §9).
- `CompiledModelCache` — EPContext artifact paths stamped `{hash}_{ep}_{ver}_{driver}` so a
  driver/EP update ignores the stale graph instead of failing `INVALID_GRAPH`.
- `SessionOptionsBuilder.ApplyFixedShapes` — `AddFreeDimensionOverrideByName` to pin dynamic axes
  (iron rule §9: DirectML dynamic axes are ~5× slower; streaming encoder = fixed chunk + fixed
  cache, offline = fixed window / length-bucket padding).
- `ModelVariantSelector` — picks `.int8` for NPU (QNN/VitisAI), `.fp16` for DirectML/CUDA, fp32 for
  CPU, honoring the manifest `providerSupport`.
- `OsCapabilities` — Win10 1809 floor for CPU/DirectML; Win11 24H2 (build 26100) floor for NPU /
  optimized EPs.

## Phase 2 — DirectML (implemented)

In the app's composition root (with `Microsoft.Windows.AI.MachineLearning`):

1. `ExecutionProviderCatalog.GetDefault().EnsureAndRegisterCertifiedAsync()` runs on a background task at
   startup (`App.OnLaunched`) to fetch certified EPs.
2. `OrtEpEnumerator.Enumerate()` reads `OrtEnv.GetEpDevices()` fresh per session and is the default device
   list for `ExecutionProviderSelector` (don't cache it).
3. For a DirectML device, `SessionOptionsBuilder.Build` appends it via `AppendExecutionProvider`; pin dynamic
   axes with `ApplyFixedShapes`.
4. Session creation is wrapped in try/catch in the pipeline builders → CPU fallback (`EpResolver` decides).
5. Prioritize the offline second-pass (SenseVoice) encoder on DML; transducer decoder/joiner stay on CPU.

## Phase 3 — NPU + optional Whisper-genai (implemented/gated)

- NPU: discovered through the same catalog; gated to `OsCapabilities.SupportsNpuEps` (build 26100+). Require a
  fully static + quantized graph (`.int8` via `ModelVariantSelector`). Falls back to CPU on non-NPU machines.
- Whisper-AR: the optional `Stt.Plugins.WhisperGenAi` project provides a `WhisperGenAiDecoder`
  (`IAsrDecoder`, `Ar` capability) over `Microsoft.ML.OnnxRuntimeGenAI`. Not in the default
  pipeline (spec §8.5): genai carries its own ORT/EP and does not inherit the unified Windows ML EP
  selection; `Microsoft.ML.OnnxRuntimeGenAI.WinML` can run on the shared Windows ML runtime.
