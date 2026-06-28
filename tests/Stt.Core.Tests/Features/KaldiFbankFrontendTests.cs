using Stt.Abstractions.Features;
using Stt.Core.Features;

namespace Stt.Core.Tests.Features;

public class KaldiFbankFrontendTests
{
    [Fact]
    public void FamilyB_Requires_Cmvn_Stats()
    {
        // LFR configured but no CMVN → reject at construction (spec §10.2 gate 3).
        var bad = new FbankOptions { NumBins = 80, Lfr = (7, 6) };
        Assert.Throws<ArgumentException>(() => new KaldiFbankFrontend(bad));
    }

    [Fact]
    public void FamilyA_OutputDim_Is_NumBins()
    {
        using var fe = new KaldiFbankFrontend(FbankOptions.FamilyA(80));
        Assert.Equal(AsrFeatureFamily.KaldiFbankPovey, fe.Family);
        Assert.Equal(80, fe.FeatureDim);
    }

    [Fact]
    public void FamilyB_OutputDim_Is_560()
    {
        using var fe = new KaldiFbankFrontend(
            FbankOptions.FamilyB(new float[560], MakeOnes(560)));
        Assert.Equal(AsrFeatureFamily.KaldiFbankLfrCmvn, fe.Family);
        Assert.Equal(560, fe.FeatureDim);
    }

    [SkippableFact]
    public void Extracts_Fbank_Of_Tone_When_Native_Present()
    {
        Skip.IfNot(KaldiNativeFbankInterop.IsAvailable,
            "kaldi_native_fbank_shim not present (see docs/native/kaldi-native-fbank.md).");

        var pcm = new float[16000];
        for (int i = 0; i < pcm.Length; i++) pcm[i] = MathF.Sin(2 * MathF.PI * 440f * i / 16000f) * 0.5f;

        using var fe = new KaldiFbankFrontend(FbankOptions.FamilyA(80));
        var feats = fe.Extract(pcm, out int frames);

        Assert.True(frames > 90 && frames < 110); // ~100 frames for 1 s @ 10 ms hop
        Assert.Equal(frames * 80, feats.Length);
        Assert.All(feats, v => Assert.True(float.IsFinite(v) && v > -30f && v < 30f)); // natural-log range
    }

    private static float[] MakeOnes(int n)
    {
        var a = new float[n];
        Array.Fill(a, 1f);
        return a;
    }
}
