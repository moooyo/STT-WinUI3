using Stt.Abstractions.Features;

namespace Stt.Core.Features;

/// <summary>
/// NeMo / GigaAM mel front-end (spec §7, Family D). Computes a librosa/Slaney mel filterbank
/// (power spectrum, Hann window, 25 ms / 10 ms) via the native knf computer, applies natural log,
/// then per-feature (per-mel-bin) normalization over time — the FilterbankFeatures recipe NVIDIA
/// NeMo (Parakeet / Canary) and GigaAM use. Output is row-major <c>[numFrames, nMels]</c>
/// (<c>nMels</c> typically 80 or 128).
/// </summary>
/// <remarks>
/// The mel/log core reuses the same verified knf code as Families A–C (only the mel scale =
/// librosa/Slaney differs). The defining NeMo property — each mel bin has zero mean / unit variance
/// over the utterance — is what downstream NeMo encoders expect; exact parity with a specific NeMo
/// export's log-zero-guard constant should be validated per model before trusting.
/// </remarks>
public sealed class NemoMelFrontend : IFeatureFrontend, IDisposable
{
    private const int SampleRate = 16000;
    private readonly int _nMels;

    public NemoMelFrontend(int nMels = 80)
    {
        if (nMels <= 0) throw new ArgumentException("nMels must be positive.", nameof(nMels));
        _nMels = nMels;
    }

    public AsrFeatureFamily Family => AsrFeatureFamily.NemoMel;
    public int FeatureDim => _nMels;

    public float[] Extract(ReadOnlySpan<float> pcm16kMono, out int numFrames)
    {
        if (!KaldiNativeFbankInterop.IsAvailable)
            throw new DllNotFoundException(
                "kaldi_native_fbank_shim native library not found. See docs/native/kaldi-native-fbank.md.");

        float[] pcm = pcm16kMono.ToArray();

        // librosa/Slaney mel: low_freq=0, high_freq=0 (→ Nyquist), Hann window, power spectrum, log.
        IntPtr h = KaldiNativeFbankInterop.knf_mel_create(
            _nMels, SampleRate, dither: 0f, snipEdges: 0, windowType: "hann",
            lowFreq: 0f, highFreq: 0f, usePower: 1, useLog: 1);
        if (h == IntPtr.Zero) throw new InvalidOperationException("knf_mel_create returned null.");

        float[] feats;
        int frames, dim;
        try
        {
            KaldiNativeFbankInterop.knf_accept(h, SampleRate, pcm, pcm.Length);
            KaldiNativeFbankInterop.knf_finish(h);
            frames = KaldiNativeFbankInterop.knf_num_frames_ready(h);
            dim = KaldiNativeFbankInterop.knf_dim(h);
            feats = new float[frames * dim];
            var frameBuf = new float[dim];
            for (int t = 0; t < frames; t++)
            {
                KaldiNativeFbankInterop.knf_get_frame(h, t, frameBuf);
                Array.Copy(frameBuf, 0, feats, t * dim, dim);
            }
        }
        finally
        {
            KaldiNativeFbankInterop.knf_destroy(h);
        }

        // Per-feature (per-mel-bin) normalization over time — the NeMo "per_feature" mode.
        PerFeatureNorm.Apply(feats, frames, dim);

        numFrames = frames;
        return feats;
    }

    public void Dispose() { }
}
