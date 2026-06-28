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
# Place kaldi_native_fbank_shim.dll on the loader path (next to Stt.Core.dll in the test output,
# or in runtimes/win-x64/native). Then:
export STT_SILERO_VAD=/path/to/silero_vad.onnx
export STT_SENSEVOICE_DIR=/path/to/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17
# Streaming Zipformer2 (icefall export with query_head_dims/value_head_dims metadata, e.g.
# sherpa-onnx-streaming-zipformer-en-2023-06-26); set REF to assert an exact sherpa-onnx match:
export STT_ZIPFORMER_DIR=/path/to/streaming-zipformer2   # encoder.onnx/decoder.onnx/joiner.onnx/tokens.txt
export STT_ZIPFORMER_WAV=$STT_ZIPFORMER_DIR/test_wavs/0.wav
export STT_ZIPFORMER_REF="AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE SQUALID QUARTER OF THE BROTHELS"
dotnet test tests/Stt.Core.Tests/Stt.Core.Tests.csproj
```

This activates the native fbank test, the Silero VAD test, the **real end-to-end transcription
test** (`SenseVoiceTranscriptionTests`) which decodes the model's bundled `test_wavs/zh.wav` +
`en.wav` and asserts the expected text, and the **streaming alignment test**
(`StreamingTransducerTests`). Verified locally: the offline chain (kaldi-fbank shim → SenseVoice
int8 → text) produces `zh.wav → 开饭时间早上九点至下午五点` and `en.wav → the tribal chieftain called
for the boy and presented him with fifty pieces of …`, and the streaming Zipformer2 transducer
produces a transcript that is an **exact match** with the sherpa-onnx reference. The fbank golden
test (`GoldenFeatureTests`) cross-checks the C# front-end against lhotse; regenerate its vectors
with `python scripts/gen_golden_features.py` (needs `numpy soundfile lhotse`).

> The Zipformer2 metadata (`query_head_dims`/`value_head_dims`/`num_heads`) is required;
> the older `model_type=zipformer` (v1) export (e.g. the 2023-02-20 bilingual model) uses a
> different state layout and is not yet supported.

> The shim build in [kaldi-native-fbank.md](kaldi-native-fbank.md) is confirmed working with the
> CMake + MSVC toolchain bundled in Visual Studio (cmake configure pulls in kissfft automatically).
