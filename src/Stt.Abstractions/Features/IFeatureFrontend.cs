namespace Stt.Abstractions.Features;

/// <summary>
/// Extracts acoustic features from 16 kHz mono PCM (spec §5.1, §7). The concrete front-end is
/// chosen by <see cref="Family"/>; parameters come from model metadata. Output is row-major
/// <c>[numFrames, FeatureDim]</c> following the model's exact convention.
/// </summary>
public interface IFeatureFrontend
{
    /// <summary>The feature family this front-end produces.</summary>
    AsrFeatureFamily Family { get; }

    /// <summary>Feature dimension per frame (e.g. 80, 128, 560).</summary>
    int FeatureDim { get; }

    /// <summary>
    /// Extract features. Returns a row-major <c>[numFrames, FeatureDim]</c> buffer; the frame
    /// count is reported via <paramref name="numFrames"/>.
    /// </summary>
    float[] Extract(ReadOnlySpan<float> pcm16kMono, out int numFrames);
}
