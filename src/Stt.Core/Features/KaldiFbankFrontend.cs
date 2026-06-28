using Stt.Abstractions.Features;

namespace Stt.Core.Features;

/// <summary>
/// kaldi-fbank front-end (spec §7.2, families A and B). Extracts log-mel filterbank features via
/// the native <c>kaldi-native-fbank</c> shim (bit-compatible with training), then applies
/// optional LFR + CMVN for Family B. Output is row-major <c>[numFrames, OutputDim]</c>.
/// </summary>
public sealed class KaldiFbankFrontend : IFeatureFrontend, IDisposable
{
    private readonly FbankOptions _opts;

    public KaldiFbankFrontend(FbankOptions opts)
    {
        _opts = opts;
        if (opts.Lfr is { } && (opts.CmvnNegMean is null || opts.CmvnInvStddev is null))
            throw new ArgumentException("Family B (LFR) requires CMVN negMean/invStddev (spec §10.2 gate 3).");
        if (opts.Lfr is { } lfr && opts.CmvnNegMean is { } nm && nm.Length != opts.NumBins * lfr.M)
            throw new ArgumentException($"CMVN length {nm.Length} != NumBins*m ({opts.NumBins * lfr.M}).");
    }

    public AsrFeatureFamily Family => _opts.Family;
    public int FeatureDim => _opts.OutputDim;

    public float[] Extract(ReadOnlySpan<float> pcm16kMono, out int numFrames)
    {
        if (!KaldiNativeFbankInterop.IsAvailable)
            throw new DllNotFoundException(
                "kaldi_native_fbank_shim native library not found. See docs/native/kaldi-native-fbank.md.");

        // The native call needs an array (not a span).
        float[] pcm = pcm16kMono.ToArray();

        IntPtr h = KaldiNativeFbankInterop.knf_create(
            _opts.NumBins, _opts.SampleRate, _opts.Dither, _opts.SnipEdges ? 1 : 0, _opts.WindowType,
            _opts.LowFreq, _opts.HighFreq, _opts.UsePower ? 1 : 0, _opts.UseLogFbank ? 1 : 0,
            _opts.NormalizeSamples ? 1 : 0);
        if (h == IntPtr.Zero) throw new InvalidOperationException("knf_create returned null.");

        try
        {
            KaldiNativeFbankInterop.knf_accept(h, _opts.SampleRate, pcm, pcm.Length);
            KaldiNativeFbankInterop.knf_finish(h);

            int frames = KaldiNativeFbankInterop.knf_num_frames_ready(h);
            int dim = KaldiNativeFbankInterop.knf_dim(h);
            var raw = new float[frames * dim];
            var frameBuf = new float[dim];
            for (int t = 0; t < frames; t++)
            {
                KaldiNativeFbankInterop.knf_get_frame(h, t, frameBuf);
                Array.Copy(frameBuf, 0, raw, t * dim, dim);
            }

            return PostProcess(raw, frames, dim, out numFrames);
        }
        finally
        {
            KaldiNativeFbankInterop.knf_destroy(h);
        }
    }

    /// <summary>Apply LFR + CMVN for Family B; Family A returns the raw fbank unchanged.</summary>
    private float[] PostProcess(float[] raw, int frames, int dim, out int numFrames)
    {
        if (_opts.Lfr is not { } lfr)
        {
            numFrames = frames;
            return raw;
        }

        float[] stacked = Lfr.Apply(raw, frames, dim, lfr.M, lfr.N, out numFrames);
        int outDim = dim * lfr.M;
        Cmvn.Apply(stacked, numFrames, outDim, _opts.CmvnNegMean!, _opts.CmvnInvStddev!);
        return stacked;
    }

    public void Dispose() { }
}
