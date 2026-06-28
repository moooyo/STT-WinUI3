# Phase 0 end-to-end setup

Phase 0 delivers offline Chinese–English transcription: speak a sentence → text. The engine is
complete; you supply three native/model artifacts (none are committed — spec D7: models are
user-supplied).

## 1. Native kaldi-fbank shim

Build `kaldi_native_fbank_shim.dll` and place it next to the app executable (or in
`runtimes/win-x64/native`). Full ABI + shim source + build steps:
[kaldi-native-fbank.md](kaldi-native-fbank.md).

Without it, `KaldiFbankFrontend.Extract` throws `DllNotFoundException` and the feature/golden tests
skip.

## 2. Silero VAD model

Download `silero_vad.onnx` (v4 or v5 — the engine auto-detects the state layout) from the
[Silero VAD repo](https://github.com/snakers4/silero-vad). Point Settings → "Silero VAD model" at it.

## 3. SenseVoice-Small model

Obtain a SenseVoice-Small ONNX export (e.g. the sherpa-onnx SenseVoice export) as a folder:

```
sense-voice-small/
├─ model.onnx            (or sense-voice.onnx / model.int8.onnx)
├─ tokens.txt
└─ manifest.json         (optional — inferred from file naming if absent)
```

The model's ONNX metadata must carry the CMVN stats (`neg_mean`, `inv_stddev`) for the Family-B
front-end; if they are missing, the engine **fails loud** at start rather than producing garbage
(spec §10.2). A minimal `manifest.json`:

```json
{
  "id": "sense-voice-small",
  "displayName": "SenseVoice-Small (zh/en/ja/ko/yue)",
  "family": "sense_voice",
  "runtime": ["offline"],
  "decoderType": "nar",
  "files": { "model": "model.onnx", "tokens": "tokens.txt" },
  "feature": { "frontEnd": "kaldi_fbank", "family": "KaldiFbankLfrCmvn", "featureDim": 560, "lfr": [7, 6], "cmvn": "metadata" },
  "capabilities": { "offlineCapable": true, "multilingual": true, "needsVad": true, "needsLfrCmvn": true },
  "languages": ["zh", "en", "ja", "ko", "yue"],
  "license": "Apache-2.0"
}
```

Place the folder under `%LocalAppData%\Stt\models\` (auto-scanned) or import it via the app's
**Models** page (Import folder…).

## 4. Run

1. Build the app (see [README](../../README.md) — VS MSBuild) and launch it via `winapp run`
   (install with `/winui-setup` if `winapp` is missing).
2. **Models** page → confirm the SenseVoice model is listed with an `offline` badge.
3. **Settings** → Mode = `OnePassOffline`, Offline model = your SenseVoice model, Silero VAD model
   = your `silero_vad.onnx`, EP = CPU → **Save**.
4. **Transcribe** → **Start**, speak a sentence, pause → the final text appears.

## Headless verification (no mic, no UI)

Drop the native shim + models and set the test env vars to activate the skippable integration
tests:

```bash
export STT_SILERO_VAD=/path/to/silero_vad.onnx
# (place kaldi_native_fbank_shim.dll on the loader path; add golden vectors per tests/golden/README.md)
dotnet test tests/Stt.Core.Tests/Stt.Core.Tests.csproj
```
