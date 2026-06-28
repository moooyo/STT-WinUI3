namespace Stt.Core.Audio;

/// <summary>
/// Converts arbitrary-rate, possibly multi-channel PCM to 16 kHz mono float (spec §5.1: frames
/// are "16k mono float"). Downmix averages channels; rate conversion decimates by an integer
/// factor with an anti-alias FIR when the source is an exact multiple of 16 kHz (e.g. 48k→16k,
/// 32k→16k), and otherwise falls back to linear interpolation.
/// </summary>
public static class Resampler
{
    public const int TargetRate = 16000;

    /// <summary>
    /// Convert interleaved PCM at <paramref name="srcRate"/>/<paramref name="srcChannels"/> to
    /// 16 kHz mono. Returns a freshly allocated array (callers on the audio thread should pool;
    /// this helper is for setup/file paths, not the hot callback).
    /// </summary>
    public static float[] ToMono16k(ReadOnlySpan<float> interleaved, int srcRate, int srcChannels)
    {
        if (srcRate <= 0) throw new ArgumentOutOfRangeException(nameof(srcRate));
        if (srcChannels <= 0) throw new ArgumentOutOfRangeException(nameof(srcChannels));

        float[] mono = Downmix(interleaved, srcChannels);

        if (srcRate == TargetRate)
            return mono;

        if (srcRate % TargetRate == 0)
            return DecimateAntiAliased(mono, srcRate / TargetRate);

        return LinearResample(mono, srcRate, TargetRate);
    }

    /// <summary>Average channels to mono. Single-channel input is returned as a copy.</summary>
    public static float[] Downmix(ReadOnlySpan<float> interleaved, int channels)
    {
        if (channels == 1)
            return interleaved.ToArray();

        int frames = interleaved.Length / channels;
        var mono = new float[frames];
        for (int f = 0; f < frames; f++)
        {
            float sum = 0f;
            int b = f * channels;
            for (int c = 0; c < channels; c++) sum += interleaved[b + c];
            mono[f] = sum / channels;
        }
        return mono;
    }

    /// <summary>Integer-factor decimation with a windowed-sinc low-pass to suppress aliasing.</summary>
    private static float[] DecimateAntiAliased(float[] mono, int factor)
    {
        if (factor == 1) return mono;

        // Low-pass at the new Nyquist (TargetRate/2). Cutoff normalized to old rate = 1/(2*factor).
        float[] taps = LowPassFir(cutoffNormalized: 0.5f / factor, numTaps: 8 * factor + 1);
        int half = taps.Length / 2;

        int outLen = (mono.Length + factor - 1) / factor;
        var outBuf = new float[outLen];
        for (int o = 0; o < outLen; o++)
        {
            int center = o * factor;
            float acc = 0f;
            for (int t = -half; t <= half; t++)
            {
                int idx = center + t;
                if (idx < 0 || idx >= mono.Length) continue;
                acc += mono[idx] * taps[t + half];
            }
            outBuf[o] = acc;
        }
        return outBuf;
    }

    /// <summary>Plain linear interpolation for non-integer rate ratios.</summary>
    private static float[] LinearResample(float[] mono, int srcRate, int dstRate)
    {
        if (mono.Length == 0) return Array.Empty<float>();
        long outLen = (long)mono.Length * dstRate / srcRate;
        if (outLen <= 0) return Array.Empty<float>();
        var outBuf = new float[outLen];
        double step = (double)srcRate / dstRate;
        for (long o = 0; o < outLen; o++)
        {
            double srcPos = o * step;
            int i0 = (int)srcPos;
            int i1 = Math.Min(i0 + 1, mono.Length - 1);
            float frac = (float)(srcPos - i0);
            outBuf[o] = mono[i0] * (1f - frac) + mono[i1] * frac;
        }
        return outBuf;
    }

    /// <summary>Windowed-sinc (Hann) low-pass FIR, unity DC gain.</summary>
    private static float[] LowPassFir(float cutoffNormalized, int numTaps)
    {
        if (numTaps % 2 == 0) numTaps++;
        var taps = new float[numTaps];
        int half = numTaps / 2;
        double sum = 0;
        for (int n = 0; n < numTaps; n++)
        {
            int k = n - half;
            double sinc = k == 0 ? 2 * cutoffNormalized
                                 : Math.Sin(2 * Math.PI * cutoffNormalized * k) / (Math.PI * k);
            double hann = 0.5 - 0.5 * Math.Cos(2 * Math.PI * n / (numTaps - 1));
            double v = sinc * hann;
            taps[n] = (float)v;
            sum += v;
        }
        // Normalize to unity DC gain.
        for (int n = 0; n < numTaps; n++) taps[n] = (float)(taps[n] / sum);
        return taps;
    }
}
