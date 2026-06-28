using Stt.Core.Features;

namespace Stt.Core.Tests.Features;

public class CmvnTests
{
    [Fact]
    public void Applies_Affine_In_Place()
    {
        // feats [1,2,3,4] dim=2 (2 frames), negMean [-1,-1], invStddev [0.5,0.5]
        var feats = new float[] { 1, 2, 3, 4 };
        Cmvn.Apply(feats, numFrames: 2, dim: 2, negMean: new float[] { -1, -1 }, invStddev: new float[] { 0.5f, 0.5f });
        Assert.Equal(new float[] { 0f, 0.5f, 1f, 1.5f }, feats);
    }

    [Fact]
    public void Rejects_Wrong_Length_Stats()
    {
        var feats = new float[4];
        Assert.Throws<ArgumentException>(() =>
            Cmvn.Apply(feats, 2, 2, negMean: new float[] { 0 }, invStddev: new float[] { 1, 1 }));
    }
}
