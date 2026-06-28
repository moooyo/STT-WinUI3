namespace Stt.Core.Features;

/// <summary>
/// Cepstral mean/variance normalization applied in place (spec §7.2): <c>x = (x + negMean) *
/// invStddev</c>. The <c>negMean</c> and <c>invStddev</c> vectors are per-feature (length =
/// feature dimension) and read from model metadata (FunASR's four keys). Applying CMVN with the
/// wrong constants — or skipping it — yields silent garbage, so they are required, never defaulted.
/// </summary>
public static class Cmvn
{
    public static void Apply(
        Span<float> feats, int numFrames, int dim, ReadOnlySpan<float> negMean, ReadOnlySpan<float> invStddev)
    {
        if (negMean.Length != dim) throw new ArgumentException($"negMean length {negMean.Length} != dim {dim}", nameof(negMean));
        if (invStddev.Length != dim) throw new ArgumentException($"invStddev length {invStddev.Length} != dim {dim}", nameof(invStddev));

        for (int t = 0; t < numFrames; t++)
        {
            int b = t * dim;
            for (int d = 0; d < dim; d++)
                feats[b + d] = (feats[b + d] + negMean[d]) * invStddev[d];
        }
    }
}
