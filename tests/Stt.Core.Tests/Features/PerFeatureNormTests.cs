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
}
