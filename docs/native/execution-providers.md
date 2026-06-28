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

## What is implemented in Core (provider-agnostic, tested)

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

## Phase 2 — DirectML (app)

In the app's composition root (with `Microsoft.WindowsAppSDK.ML`):

1. Start `ExecutionProviderCatalog.GetDefault().EnsureAndRegisterCertifiedAsync()` on a background
   thread (progress UI) to fetch certified EPs.
2. Enumerate `OrtEnv.GetEpDevices()` fresh per session and filter by `EpName` + `HardwareDevice.Type`
   (don't cache the device list); inject this enumerator into `ExecutionProviderSelector`.
3. For a DirectML device, append the DML EP to the `SessionOptions` and call
   `SessionOptionsBuilder.ApplyFixedShapes(...)` with the model's dynamic dim names.
4. Wrap session creation in try/catch → fall back to CPU (`EpResolver` already decides this).
5. Prioritize the offline second-pass (SenseVoice) encoder on DML; transducer decoder/joiner stay
   on CPU (small matrices).

## Phase 3 — NPU + optional Whisper-genai (app/plugin)

- NPU: gate to `OsCapabilities.SupportsNpuEps` (build 26100+); use `SetEpSelectionPolicy(PREFER_NPU)`
  or filter `GetEpDevices()` for `HardwareDevice.Type == NPU`. Require a fully static + quantized
  graph with no `Loop`/`If` (QNN has no dynamic shapes). Use `.int8` variants
  (`ModelVariantSelector`). Validate QDQ accuracy against the float baseline (spec §17).
- Whisper-AR: the optional `Stt.Plugins.WhisperGenAi` project provides a `WhisperGenAiDecoder`
  (`IAsrDecoder`, `Ar` capability) over `Microsoft.ML.OnnxRuntimeGenAI`. Not in the default
  pipeline (spec §8.5): genai carries its own ORT/EP and does not inherit the unified Windows ML EP
  selection; `Microsoft.ML.OnnxRuntimeGenAI.WinML` can run on the shared Windows ML runtime.
