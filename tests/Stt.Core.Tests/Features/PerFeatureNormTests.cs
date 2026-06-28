using Stt.Core.Features;

namespace Stt.Core.Tests.Features;

public class PerFeatureNormTests
{
    [Fact]
    public void Normalizes_Each_Feature_To_Zero_Mean_Unit_Std()
    {
        // 3 frames, 1 dim, ramp [1,2,3] → mean 2, population std sqrt(2/3)
        var feats = new float[] { 1, 2, 3 };
        PerFeatureNorm.Apply(feats, numFrames: 3, dim: 1);

        float mean = (feats[0] + feats[1] + feats[2]) / 3f;
        Assert.InRange(mean, -1e-4f, 1e-4f);

        double var = (feats[0] * feats[0] + feats[1] * feats[1] + feats[2] * feats[2]) / 3.0;
        Assert.InRange(var, 0.99, 1.01);
        Assert.True(feats[0] < feats[1] && feats[1] < feats[2]); // order preserved
    }

    [Fact]
    public void Constant_Feature_Does_Not_NaN()
    {
        var feats = new float[] { 5, 5, 5 };
        PerFeatureNorm.Apply(feats, 3, 1);
        Assert.All(feats, f => Assert.False(float.IsNaN(f)));
    }

    [Fact]
    public void Constant_Feature_Normalizes_To_Near_Zero()
    {
        // mean==value, diff==0 → output ~0 (eps guard prevents div-by-zero).
        var feats = new float[] { 5, 5, 5, 5 };
        PerFeatureNorm.Apply(feats, 4, 1);
        Assert.All(feats, f => Assert.InRange(f, -1e-3f, 1e-3f));
    }

    [Fact]
    public void Affine_Transform_Of_Input_Yields_Same_Normalization()
    {
        // For y = a*x + b (a > 0), normalized(y) == normalized(x): the normalizer is affine-invariant.
        var x = new float[] { -3, 1, 4, 7, 9 };
        var y = new float[x.Length];
        const float a = 2.5f, b = -10f;
        for (int i = 0; i < x.Length; i++) y[i] = a * x[i] + b;

        PerFeatureNorm.Apply(x, x.Length, dim: 1);
        PerFeatureNorm.Apply(y, y.Length, dim: 1);

        for (int i = 0; i < x.Length; i++)
            Assert.InRange(y[i] - x[i], -1e-3f, 1e-3f);
    }

    [Fact]
    public void MultiDim_Normalizes_Each_Column_Independently()
    {
        // 4 frames, dim=3; each column has a distinct scale+offset and must end ~0 mean / ~unit var,
        // with no cross-column leakage.
        const int frames = 4, dim = 3;
        var feats = new float[frames * dim];
        for (int t = 0; t < frames; t++)
        {
            feats[t * dim + 0] = 1 + t;            // 1,2,3,4
            feats[t * dim + 1] = 100 - 10 * t;     // 100,90,80,70
            feats[t * dim + 2] = -5 + 0.5f * t;    // -5,-4.5,-4,-3.5
        }
        PerFeatureNorm.Apply(feats, frames, dim);

        for (int d = 0; d < dim; d++)
        {
            double mean = 0, var = 0;
            for (int t = 0; t < frames; t++) mean += feats[t * dim + d];
            mean /= frames;
            for (int t = 0; t < frames; t++) { double diff = feats[t * dim + d] - mean; var += diff * diff; }
            var /= frames;
            Assert.InRange(mean, -1e-4, 1e-4);
            Assert.InRange(var, 0.99, 1.01);
        }
    }

    [Fact]
    public void KnownVector_Uses_Population_Not_Sample_Variance()
    {
        // col0 = [1,2,3] over 3 frames, dim=2. Population std = sqrt(2/3); normalized[0] = (1-2)/sqrt(2/3)
        // = -sqrt(3/2) ≈ -1.224745. Sample variance (÷N-1=2) would give std=1 → -1.0, so this pins the
        // population-variance convention (m2/numFrames, PerFeatureNorm.cs:37).
        var feats = new float[] { 1, 0, 2, 0, 3, 0 }; // interleaved: col0 ramps, col1 constant 0
        PerFeatureNorm.Apply(feats, numFrames: 3, dim: 2);
        Assert.InRange(feats[0], -1.2247f - 1e-3f, -1.2247f + 1e-3f);
    }

    [Fact]
    public void NonPositive_NumFrames_Leaves_Buffer_Untouched()
    {
        var feats = new float[] { 7, 8, 9 };
        PerFeatureNorm.Apply(feats, numFrames: 0, dim: 1); // early return at PerFeatureNorm.cs:12
        Assert.Equal(new float[] { 7, 8, 9 }, feats);
    }
}
