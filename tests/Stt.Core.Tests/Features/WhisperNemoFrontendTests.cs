using Stt.Abstractions.Features;
using Stt.Core.Features;

namespace Stt.Core.Tests.Features;

/// <summary>Unit + native checks for the Family C (Whisper) and Family D (NeMo) front-ends.</summary>
public class WhisperNemoFrontendTests
{
    [Fact]
    public void Whisper_Reports_Family_And_Dim()
    {
        using var fe = new WhisperMelFrontend(128);
        Assert.Equal(AsrFeatureFamily.WhisperLogMel, fe.Family);
        Assert.Equal(128, fe.FeatureDim);
    }

    [Theory]
    [InlineData(80)]    // Whisper ≤ large-v2
    [InlineData(128)]   // Whisper large-v3
    [InlineData(96)]    // a non-standard (e.g. Qwen-style) bin count is metadata-driven, not rejected
    public void Whisper_Accepts_MetadataDriven_NMels(int n)
    {
        using var fe = new WhisperMelFrontend(n);
        Assert.Equal(n, fe.FeatureDim);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    [InlineData(9999)]  // fail loud on nonsensical values rather than silently guessing 80
    public void Whisper_Rejects_Nonsensical_NMels(int n) =>
        Assert.Throws<ArgumentException>(() => new WhisperMelFrontend(n));

    [Fact]
    public void Nemo_Reports_Family_And_Dim()
    {
        using var fe = new NemoMelFrontend(80);
        Assert.Equal(AsrFeatureFamily.NemoMel, fe.Family);
        Assert.Equal(80, fe.FeatureDim);
    }

    [SkippableFact]
    public void Nemo_Output_Is_PerFeature_Normalized()
    {
        Skip.IfNot(KaldiNativeFbankInterop.IsAvailable, "native fbank shim not present.");

        // 1 s of mixed tones so every mel bin sees some energy.
        var pcm = new float[16000];
        for (int i = 0; i < pcm.Length; i++)
            pcm[i] = (MathF.Sin(2 * MathF.PI * 220f * i / 16000f)
                      + 0.6f * MathF.Sin(2 * MathF.PI * 1500f * i / 16000f)
                      + 0.4f * MathF.Sin(2 * MathF.PI * 4000f * i / 16000f)) * 0.3f;

        using var fe = new NemoMelFrontend(80);
        float[] feats = fe.Extract(pcm, out int frames);

        Assert.True(frames > 90 && frames < 110);
        Assert.Equal(frames * 80, feats.Length);
        Assert.All(feats, v => Assert.True(float.IsFinite(v)));

        // Defining NeMo "per_feature" property: each mel bin ~zero mean, ~unit std over time.
        for (int d = 0; d < 80; d++)
        {
            double mean = 0;
            for (int t = 0; t < frames; t++) mean += feats[t * 80 + d];
            mean /= frames;
            double var = 0;
            for (int t = 0; t < frames; t++) { double x = feats[t * 80 + d] - mean; var += x * x; }
            var /= frames;
            Assert.True(Math.Abs(mean) < 1e-3, $"bin {d} mean {mean} not ~0");
            // std ~1, but the epsilon guard in PerFeatureNorm pulls low-variance bins slightly under.
            Assert.True(Math.Sqrt(var) > 0.85 && Math.Sqrt(var) <= 1.01, $"bin {d} std {Math.Sqrt(var)} not ~1");
        }
    }
}
