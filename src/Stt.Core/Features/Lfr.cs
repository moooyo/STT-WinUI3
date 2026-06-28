namespace Stt.Core.Features;

/// <summary>
/// Low Frame Rate stacking as used by FunASR Paraformer / SenseVoice (spec §7.2). Given
/// row-major features <c>[T, D]</c>, produces <c>[ceil(T/n), D*m]</c> by replicating the first
/// frame <c>(m-1)/2</c> times at the front, then stacking <c>m</c> consecutive frames stepping
/// by <c>n</c>; the final window is padded by replicating the last frame. SenseVoice uses
/// <c>m=7, n=6</c> → 80×7 = 560.
/// </summary>
public static class Lfr
{
    public static float[] Apply(
        ReadOnlySpan<float> feats, int numFrames, int featDim, int lfrM, int lfrN, out int outFrames)
    {
        if (lfrM <= 0 || lfrN <= 0) throw new ArgumentOutOfRangeException(nameof(lfrM));
        if (numFrames <= 0) { outFrames = 0; return Array.Empty<float>(); }

        int leftPad = (lfrM - 1) / 2;
        int paddedFrames = numFrames + leftPad;

        outFrames = (int)Math.Ceiling(numFrames / (double)lfrN);
        int outDim = featDim * lfrM;
        var outBuf = new float[outFrames * outDim];

        for (int i = 0; i < outFrames; i++)
        {
            int baseFrame = i * lfrN;        // index into the padded frame list
            for (int k = 0; k < lfrM; k++)
            {
                int padIdx = baseFrame + k;
                if (padIdx >= paddedFrames) padIdx = paddedFrames - 1; // replicate last frame

                int real = padIdx - leftPad;  // index into original feats
                if (real < 0) real = 0;        // front replication of frame0
                else if (real >= numFrames) real = numFrames - 1;

                feats.Slice(real * featDim, featDim)
                     .CopyTo(outBuf.AsSpan(i * outDim + k * featDim, featDim));
            }
        }
        return outBuf;
    }
}
