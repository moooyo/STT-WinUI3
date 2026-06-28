using Stt.Core.Features;

namespace Stt.Core.Tests.Features;

public class LfrTests
{
    [Fact]
    public void Funasr_Semantics_5Frames_M3_N2()
    {
        // frames f_k = [2k, 2k+1], dim=2, T=5
        var feats = new float[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        var outBuf = Lfr.Apply(feats, numFrames: 5, featDim: 2, lfrM: 3, lfrN: 2, out int outFrames);

        Assert.Equal(3, outFrames);             // ceil(5/2)
        Assert.Equal(3 * 6, outBuf.Length);     // outDim = 2*3 = 6

        // leftPad=(3-1)/2=1 → padded = [f0,f0,f1,f2,f3,f4]
        // row0 = [f0,f0,f1] = 0,1, 0,1, 2,3
        // row1 = [f1,f2,f3] = 2,3, 4,5, 6,7
        // row2 = [f3,f4,f4] = 6,7, 8,9, 8,9
        Assert.Equal(new float[] { 0, 1, 0, 1, 2, 3 }, outBuf[0..6]);
        Assert.Equal(new float[] { 2, 3, 4, 5, 6, 7 }, outBuf[6..12]);
        Assert.Equal(new float[] { 6, 7, 8, 9, 8, 9 }, outBuf[12..18]);
    }

    [Fact]
    public void SenseVoice_Geometry_80x7_Yields_560()
    {
        var feats = new float[20 * 80]; // 20 frames of 80-dim fbank
        var outBuf = Lfr.Apply(feats, 20, 80, lfrM: 7, lfrN: 6, out int outFrames);
        Assert.Equal((int)Math.Ceiling(20 / 6.0), outFrames); // 4
        Assert.Equal(outFrames * 560, outBuf.Length);
    }
}
