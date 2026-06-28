// Thin C-ABI shim over kaldi-native-fbank, matching src/Stt.Core/Features/KaldiNativeFbankInterop.cs.
// A small polymorphic base (IFeat) lets the same accept/finish/num-frames/dim/get-frame entry
// points drive any knf computer — povey fbank (Family A/B), librosa/Slaney mel (Family D, NeMo),
// or the Whisper log-mel computer (Family C) — so adding a family is just one more create function.
#include "kaldi-native-fbank/csrc/online-feature.h"
#include "kaldi-native-fbank/csrc/feature-fbank.h"
#include "kaldi-native-fbank/csrc/whisper-feature.h"
#include <cstring>
#include <string>
using namespace knf;

#define KNF_API extern "C" __declspec(dllexport)

namespace {
// Type-erased feature extractor: one virtual surface over the knf online computers.
struct IFeat {
  virtual ~IFeat() = default;
  virtual void Accept(float sr, const float* s, int n) = 0;
  virtual void Finish() = 0;
  virtual int NumFrames() = 0;
  virtual int Dim() = 0;
  virtual void GetFrame(int i, float* out) = 0;
};

template <class T, class Opts>
struct FeatWrap : IFeat {
  T impl;
  explicit FeatWrap(const Opts& o) : impl(o) {}
  void Accept(float sr, const float* s, int n) override { impl.AcceptWaveform(sr, s, n); }
  void Finish() override { impl.InputFinished(); }
  int NumFrames() override { return impl.NumFramesReady(); }
  int Dim() override { return impl.Dim(); }
  void GetFrame(int i, float* out) override {
    std::memcpy(out, impl.GetFrame(i), static_cast<size_t>(impl.Dim()) * sizeof(float));
  }
};

void ConfigFrame(FrameExtractionOptions& f, float sr, float dither, int snip_edges,
                 const char* window_type) {
  f.samp_freq = sr;
  f.dither = dither;
  f.snip_edges = snip_edges != 0;
  f.window_type = window_type ? std::string(window_type) : "povey";
}
}  // namespace

// Family A/B: povey/HTK kaldi-fbank. Signature is unchanged from the original shim.
KNF_API void* knf_create(int num_bins, float sample_rate, float dither, int snip_edges,
                         const char* window_type, float low_freq, float high_freq,
                         int use_power, int use_log, int normalize_samples) {
  FbankOptions opts;
  ConfigFrame(opts.frame_opts, sample_rate, dither, snip_edges, window_type);
  opts.mel_opts.num_bins = num_bins;
  opts.mel_opts.low_freq = low_freq;
  opts.mel_opts.high_freq = high_freq;
  opts.use_power = use_power != 0;
  opts.use_log_fbank = use_log != 0;
  (void)normalize_samples;
  return new FeatWrap<OnlineFbank, FbankOptions>(opts);
}

// Family D (NeMo / GigaAM): librosa/Slaney mel filterbank. low_freq defaults to 0 for librosa.
KNF_API void* knf_mel_create(int num_bins, float sample_rate, float dither, int snip_edges,
                             const char* window_type, float low_freq, float high_freq,
                             int use_power, int use_log) {
  FbankOptions opts;
  ConfigFrame(opts.frame_opts, sample_rate, dither, snip_edges, window_type);
  opts.mel_opts.num_bins = num_bins;
  opts.mel_opts.low_freq = low_freq;
  opts.mel_opts.high_freq = high_freq;
  opts.mel_opts.is_librosa = true;
  opts.mel_opts.norm = "slaney";
  opts.mel_opts.use_slaney_mel_scale = true;
  opts.use_power = use_power != 0;
  opts.use_log_fbank = use_log != 0;
  return new FeatWrap<OnlineFbank, FbankOptions>(opts);
}

// Family C (Whisper / Qwen): Whisper log-mel computer. dim = 80 (large-v2 and earlier) or 128
// (large-v3). Output is LINEAR mel energy per frame; the caller applies log10 + Whisper's global
// dynamic-range clamp/normalize. knf's WhisperFeatureComputer resets frame_opts internally to the
// canonical Whisper framing (Hann, 25 ms/10 ms, n_fft=400), so only the mel dim matters here.
KNF_API void* knf_whisper_create(int dim, float sample_rate) {
  (void)sample_rate;  // Whisper is fixed 16 kHz; frame_opts are set inside the computer.
  return new FeatWrap<OnlineWhisperFbank, WhisperFeatureOptions>(WhisperFeatureOptions({}, dim));
}

// Polymorphic ops — work on any handle from the create functions above.
KNF_API void knf_accept(void* h, float sr, const float* s, int n) { static_cast<IFeat*>(h)->Accept(sr, s, n); }
KNF_API void knf_finish(void* h) { static_cast<IFeat*>(h)->Finish(); }
KNF_API int  knf_num_frames_ready(void* h) { return static_cast<IFeat*>(h)->NumFrames(); }
KNF_API int  knf_dim(void* h) { return static_cast<IFeat*>(h)->Dim(); }
KNF_API void knf_get_frame(void* h, int i, float* out) { static_cast<IFeat*>(h)->GetFrame(i, out); }
KNF_API void knf_destroy(void* h) { delete static_cast<IFeat*>(h); }
