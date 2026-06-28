using Stt.Core.Audio;

namespace Stt.Core.Tests.Audio;

public class ResamplerTests
{
    [Fact]
    public void Passthrough_At_16k_Mono_Is_Identity()
    {
        var src = new float[] { 0.1f, -0.2f, 0.3f, -0.4f };
        var outBuf = Resampler.ToMono16k(src, 16000, 1);
        Assert.Equal(src, outBuf);
    }

    [Fact]
    public void Stereo_Downmix_Averages_Channels()
    {
        // L/R interleaved: (1,3),(2,4) → mono (2,3)
        var src = new float[] { 1f, 3f, 2f, 4f };
        var mono = Resampler.Downmix(src, 2);
        Assert.Equal(new float[] { 2f, 3f }, mono);
    }

    [Fact]
    public void Decimate_48k_To_16k_Produces_Third_Length()
    {
        // 48k → 16k is integer factor 3.
        int srcLen = 4800; // 0.1 s at 48k
        var sine = new float[srcLen];
        for (int i = 0; i < srcLen; i++) sine[i] = MathF.Sin(2 * MathF.PI * 220f * i / 48000f);
        var outBuf = Resampler.ToMono16k(sine, 48000, 1);
        Assert.InRange(outBuf.Length, 1599, 1601); // ~1600
        Assert.All(outBuf, v => Assert.InRange(v, -1.5f, 1.5f));
    }

    [Fact]
    public void NonInteger_Rate_Uses_Linear_And_Scales_Length()
    {
        var src = new float[44100]; // 1 s at 44.1k
        for (int i = 0; i < src.Length; i++) src[i] = MathF.Sin(2 * MathF.PI * 100f * i / 44100f);
        var outBuf = Resampler.ToMono16k(src, 44100, 1);
        Assert.InRange(outBuf.Length, 15990, 16010); // ~16000
    }
}
