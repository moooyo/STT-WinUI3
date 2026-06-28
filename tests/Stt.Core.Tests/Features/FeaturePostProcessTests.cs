using Stt.Core.Features;

namespace Stt.Core.Tests.Features;

/// <summary>
/// Dependency-free numeric coverage of the feature post-processing chain (LFR stacking → CMVN),
/// the exact composition <see cref="KaldiFbankFrontend"/> runs after the native fbank step. The
/// full fbank-vs-Python golden (plan Task 0.16) stays blocked on the native shim + reference
/// vectors; this exercises the pure-C# math the golden would otherwise be the only path to.
/// </summary>
public class FeaturePostProcessTests
{
    [Fact]
    public void Lfr_Then_Cmvn_Produces_560Dim_ZeroMean_UnitVar()
    {
        // Synthetic raw fbank [T,80] with per-(t,d) variation so stacked columns vary across rows.
        const int T = 40, D = 80, m = 7, n = 6;
        var raw = new float[T * D];
        for (int t = 0; t < T; t++)
            for (int d = 0; d < D; d++)
                raw[t * D + d] = 0.5f * t + 0.01f * d;

        float[] stacked = Lfr.Apply(raw, T, D, lfrM: m, lfrN: n, out int outFrames);
        int outDim = D * m;
        Assert.Equal(560, outDim);                       // SenseVoice geometry 80×7
        Assert.Equal(outFrames * outDim, stacked.Length);

        // Derive CMVN constants that zero-mean / unit-var each of the 560 columns (population stats).
        var negMean = new float[outDim];
        var invStd = new float[outDim];
        for (int d = 0; d < outDim; d++)
        {
            double mean = 0;
            for (int t = 0; t < outFrames; t++) mean += stacked[t * outDim + d];
            mean /= outFrames;
            double var = 0;
            for (int t = 0; t < outFrames; t++) { double diff = stacked[t * outDim + d] - mean; var += diff * diff; }
            var /= outFrames;
            double std = Math.Sqrt(var);
            negMean[d] = (float)(-mean);
            invStd[d] = std > 1e-6 ? (float)(1.0 / std) : 1f;
        }

        Cmvn.Apply(stacked, outFrames, outDim, negMean, invStd);

        for (int d = 0; d < outDim; d++)
        {
            double mean = 0;
            for (int t = 0; t < outFrames; t++) mean += stacked[t * outDim + d];
            mean /= outFrames;
            Assert.InRange(mean, -1e-3, 1e-3);          // CMVN removed the column mean

            double var = 0;
            for (int t = 0; t < outFrames; t++) { double diff = stacked[t * outDim + d] - mean; var += diff * diff; }
            var /= outFrames;
            // Columns that had real variance must now be unit variance; pure-constant tail columns stay ~0.
            if (var > 1e-4) Assert.InRange(var, 0.97, 1.03);
        }

        Assert.All(stacked, f => Assert.False(float.IsNaN(f) || float.IsInfinity(f)));
    }

    [Fact]
    public void Cmvn_Is_The_Affine_Map_It_Documents()
    {
        // x' = (x + negMean) * invStddev applied per feature dimension (Cmvn.cs:21).
        var feats = new float[] { 2, 10, 4, 20 }; // 2 frames, dim=2
        var negMean = new float[] { -1f, -5f };
        var invStd = new float[] { 0.5f, 0.1f };
        Cmvn.Apply(feats, numFrames: 2, dim: 2, negMean, invStd);

        Assert.Equal((2 - 1) * 0.5f, feats[0], 5);
        Assert.Equal((10 - 5) * 0.1f, feats[1], 5);
        Assert.Equal((4 - 1) * 0.5f, feats[2], 5);
        Assert.Equal((20 - 5) * 0.1f, feats[3], 5);
    }

    [Fact]
    public void Cmvn_Rejects_Mismatched_Stat_Lengths()
    {
        var feats = new float[] { 1, 2, 3, 4 };
        Assert.Throws<ArgumentException>(() =>
            Cmvn.Apply(feats, numFrames: 2, dim: 2, negMean: new float[] { 0 }, invStddev: new float[] { 1, 1 }));
    }
}
