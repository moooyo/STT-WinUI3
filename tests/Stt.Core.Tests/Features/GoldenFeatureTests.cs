using Stt.Core.Audio;
using Stt.Core.Features;

namespace Stt.Core.Tests.Features;

/// <summary>
/// Golden numeric validation of the C# fbank front-end against Python references (spec §7.3).
/// Skips unless both the native fbank shim AND the committed golden vectors are present. Asserts
/// element-wise max-abs &lt; 1e-3 and mean-abs &lt; 1e-4. Regenerate goldens via
/// scripts/gen_golden_features.py (see tests/golden/README.md).
/// </summary>
public class GoldenFeatureTests
{
    private static string GoldenDir =>
        Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "golden");

    [SkippableFact]
    public void FamilyA_Matches_Python_Reference()
    {
        Skip.IfNot(KaldiNativeFbankInterop.IsAvailable, "native fbank shim not present.");
        string wav = Path.Combine(GoldenDir, "input.wav");
        string bin = Path.Combine(GoldenDir, "featsA.bin");
        string shape = Path.Combine(GoldenDir, "featsA.shape");
        Skip.IfNot(File.Exists(wav) && File.Exists(bin) && File.Exists(shape),
            "Family A golden vectors not present (run scripts/gen_golden_features.py).");

        var (goldT, goldDim, gold) = LoadGolden(bin, shape);
        var audio = WavIo.ReadPcm(wav);
        float[] mono = Resampler.ToMono16k(audio.Interleaved, audio.SampleRate, audio.Channels);

        using var fe = new KaldiFbankFrontend(FbankOptions.FamilyA(goldDim));
        float[] feats = fe.Extract(mono, out int frames);

        AssertClose(gold, goldT, goldDim, feats, frames, fe.FeatureDim);
    }

    private static (int T, int Dim, float[] Data) LoadGolden(string bin, string shape)
    {
        string[] parts = File.ReadAllText(shape).Trim().Split(' ', StringSplitOptions.RemoveEmptyEntries);
        int t = int.Parse(parts[0]), dim = int.Parse(parts[1]);
        byte[] bytes = File.ReadAllBytes(bin);
        var data = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, data, 0, bytes.Length);
        return (t, dim, data);
    }

    private static void AssertClose(float[] gold, int gT, int gDim, float[] got, int frames, int dim)
    {
        Assert.Equal(gDim, dim);
        int t = Math.Min(gT, frames);
        double maxAbs = 0, sumAbs = 0;
        int n = 0;
        for (int i = 0; i < t; i++)
            for (int d = 0; d < dim; d++)
            {
                double diff = Math.Abs(gold[i * gDim + d] - got[i * dim + d]);
                maxAbs = Math.Max(maxAbs, diff);
                sumAbs += diff; n++;
            }
        double meanAbs = n > 0 ? sumAbs / n : 0;
        Assert.True(maxAbs < 1e-3, $"max-abs {maxAbs} >= 1e-3");
        Assert.True(meanAbs < 1e-4, $"mean-abs {meanAbs} >= 1e-4");
    }
}
