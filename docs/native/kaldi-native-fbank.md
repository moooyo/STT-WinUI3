# kaldi-native-fbank native shim

The `KaldiFbankFrontend` (spec §7.2, decision **D6**) extracts log-mel filterbank features by
P/Invoking a small C ABI shim that wraps the C++ [`kaldi-native-fbank`](https://github.com/csukuangfj/kaldi-native-fbank)
(`knf::OnlineFbank`). We own the shim so the ABI is stable and bit-compatible with the
training-time fbank. Apache-2.0. Built per RID into `runtimes/win-x64/native` and
`runtimes/win-arm64/native`, so it lands next to the executable in an unpackaged self-contained
build (spec §13).

## ABI (must match `KaldiNativeFbankInterop.cs`)

```c
// kaldi_native_fbank_shim.h
#ifdef __cplusplus
extern "C" {
#endif

// Create an online fbank extractor. Returns an opaque handle (NULL on failure).
void* knf_create(int num_bins, float sample_rate, float dither, int snip_edges,
                 const char* window_type, float low_freq, float high_freq,
                 int use_power, int use_log, int normalize_samples);

// Feed waveform samples (float). May be called repeatedly for streaming.
void  knf_accept(void* handle, float sample_rate, const float* samples, int count);

// Signal end of input (flush trailing frames).
void  knf_finish(void* handle);

// Number of frames ready to read.
int   knf_num_frames_ready(void* handle);

// Feature dimension per frame (== num_bins).
int   knf_dim(void* handle);

// Copy frame `frame_index` (length == knf_dim) into `out` (caller-allocated).
void  knf_get_frame(void* handle, int frame_index, float* out);

// Free the handle.
void  knf_destroy(void* handle);

#ifdef __cplusplus
}
#endif
```

## Shim implementation (`kaldi_native_fbank_shim.cc`)

```cpp
#include "kaldi-native-fbank/csrc/online-feature.h"
#include "kaldi-native-fbank/csrc/feature-fbank.h"
#include <cstring>
#include <string>
using namespace knf;

extern "C" void* knf_create(int num_bins, float sample_rate, float dither, int snip_edges,
                            const char* window_type, float low_freq, float high_freq,
                            int use_power, int use_log, int normalize_samples) {
  FbankOptions opts;
  opts.frame_opts.samp_freq = sample_rate;
  opts.frame_opts.dither = dither;
  opts.frame_opts.snip_edges = snip_edges != 0;
  opts.frame_opts.window_type = window_type ? std::string(window_type) : "povey";
  // kaldi-native-fbank treats samples in [-32768,32768] by default; when the caller already
  // has [-1,1] audio, normalize_samples==1 means "do not rescale" — knf exposes this via
  // FrameExtractionOptions; emulate by leaving samples as-is (our callers always pass [-1,1]).
  opts.mel_opts.num_bins = num_bins;
  opts.mel_opts.low_freq = low_freq;
  opts.mel_opts.high_freq = high_freq;   // <=0 is interpreted relative to Nyquist by knf
  opts.use_power = use_power != 0;
  opts.use_log_fbank = use_log != 0;
  (void)normalize_samples;               // our pipeline feeds [-1,1]; scale here if you feed int16
  return new OnlineFbank(opts);
}

extern "C" void knf_accept(void* h, float sr, const float* s, int n) {
  reinterpret_cast<OnlineFbank*>(h)->AcceptWaveform(sr, s, n);
}
extern "C" void knf_finish(void* h) { reinterpret_cast<OnlineFbank*>(h)->InputFinished(); }
extern "C" int  knf_num_frames_ready(void* h) { return reinterpret_cast<OnlineFbank*>(h)->NumFramesReady(); }
extern "C" int  knf_dim(void* h) { return reinterpret_cast<OnlineFbank*>(h)->Dim(); }
extern "C" void knf_get_frame(void* h, int i, float* out) {
  const float* f = reinterpret_cast<OnlineFbank*>(h)->GetFrame(i);
  std::memcpy(out, f, reinterpret_cast<OnlineFbank*>(h)->Dim() * sizeof(float));
}
extern "C" void knf_destroy(void* h) { delete reinterpret_cast<OnlineFbank*>(h); }
```

> **normalize_samples note (spec §7.3):** if you change the pipeline to feed int16-scaled audio,
> handle the ×32768 here; a constant feature offset of ≈10.4 is the classic symptom of getting
> this switch wrong. Our managed pipeline always feeds `[-1,1]`.

## Build (Windows, x64)

```powershell
git clone https://github.com/csukuangfj/kaldi-native-fbank
cmake -S kaldi-native-fbank -B build-x64 -A x64 -DKNF_BUILD_PYTHON=OFF -DBUILD_SHARED_LIBS=ON
cmake --build build-x64 --config Release
# Then compile the shim against the knf static/shared lib and headers, producing
# kaldi_native_fbank_shim.dll. Copy to:
#   src/Stt.App/runtimes/win-x64/native/kaldi_native_fbank_shim.dll
# Repeat with -A ARM64 for runtimes/win-arm64/native.
```

When the DLL is absent, `KaldiFbankFrontend.Extract` throws `DllNotFoundException` and the
golden/feature tests skip (`KaldiNativeFbankInterop.IsAvailable == false`).

## Extended shim — Families C and D

The maintained shim source lives in [`native/shim/`](../../native/shim/) (`kaldi_native_fbank_shim.cc`
+ `CMakeLists.txt`); the snippet above is the original A/B version. The current shim wraps all knf
online computers behind one polymorphic `IFeat` surface, so the same `knf_accept/finish/
num_frames_ready/dim/get_frame/destroy` entry points drive three create functions:

| Create | knf computer | Family |
|---|---|---|
| `knf_create(...)` | `OnlineFbank` (povey/HTK mel) | A / B |
| `knf_mel_create(...)` | `OnlineFbank` with `is_librosa`/Slaney mel | D (NeMo / GigaAM) |
| `knf_whisper_create(dim, sr)` | `OnlineWhisperFbank` | C (Whisper / Qwen), dim 80 or 128 |

`knf_whisper_create` returns LINEAR mel energy; `WhisperMelFrontend` (C#) applies `log10` + Whisper's
global dynamic-range normalize and pads/trims to the fixed 30 s (3000-frame) window. `knf_mel_create`
output is consumed by `NemoMelFrontend`, which applies natural log + per-feature normalization.
Rebuild after editing: `cmake -S native/shim -B build -A x64 && cmake --build build --config Release`.
