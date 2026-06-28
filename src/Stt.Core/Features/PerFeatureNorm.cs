namespace Stt.Core.Features;

/// <summary>
/// Per-feature (per-dimension) normalization over time, as NeMo/GigaAM models expect (spec §7.1
/// family D): for each feature dimension, subtract its mean and divide by its standard deviation
/// computed across all frames. Applied in place. A small epsilon guards silent/constant features.
/// </summary>
public static class PerFeatureNorm
{
    public static void Apply(Span<float> feats, int numFrames, int dim, float epsilon = 1e-5f)
    {
        if (numFrames <= 0) return;

        Span<double> mean = new double[dim];
        Span<double> m2 = new double[dim];

        for (int t = 0; t < numFrames; t++)
        {
            int b = t * dim;
            for (int d = 0; d < dim; d++) mean[d] += feats[b + d];
        }
        for (int d = 0; d < dim; d++) mean[d] /= numFrames;

        for (int t = 0; t < numFrames; t++)
        {
            int b = t * dim;
            for (int d = 0; d < dim; d++)
            {
                double diff = feats[b + d] - mean[d];
                m2[d] += diff * diff;
            }
        }

        Span<float> invStd = new float[dim];
        for (int d = 0; d < dim; d++)
        {
            double var = m2[d] / numFrames;
            invStd[d] = (float)(1.0 / Math.Sqrt(var + epsilon));
        }

        for (int t = 0; t < numFrames; t++)
        {
            int b = t * dim;
            for (int d = 0; d < dim; d++)
                feats[b + d] = (float)((feats[b + d] - mean[d]) * invStd[d]);
        }
    }
}
