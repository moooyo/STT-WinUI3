namespace Stt.Core.Models;

/// <summary>How a model's feature input tensor is laid out (spec §10.1).</summary>
public enum FeatureLayout
{
    /// <summary>Unknown / not a 3-D feature input.</summary>
    Unknown,

    /// <summary>fbank / LFR: <c>[N, T, C]</c> — feature dim is the last axis, time (axis 1) is dynamic.</summary>
    TimeLast,

    /// <summary>Whisper: <c>[N, n_mels, 3000]</c> — mel count is the middle axis, time (axis 2) is fixed 3000.</summary>
    MelMid
}

/// <summary>
/// The signals extracted from a model needed to detect its feature family and validate it
/// (spec §10.1): the declared <c>model_type</c>, the arbiter feature dimension, optional declared
/// mel count, the input layout, and the raw metadata map for parameter gates.
/// </summary>
public sealed record ModelProbe
{
    /// <summary>Declared architecture, e.g. "zipformer2", "sense_voice_ctc", "whisper", "EncDecCTC".</summary>
    public string ModelType { get; init; } = "";

    /// <summary>The arbiter: the feature input tensor's feature dimension (80 / 128 / 560 / 40 ...).</summary>
    public int FeatureDim { get; init; }

    /// <summary>Declared n_mels if present in metadata; otherwise null.</summary>
    public int? NMels { get; init; }

    /// <summary>Detected input layout.</summary>
    public FeatureLayout Layout { get; init; } = FeatureLayout.Unknown;

    /// <summary>Encoder vocab size (for token-count validation), or 0 if unknown.</summary>
    public int VocabSize { get; init; }

    /// <summary>Raw custom metadata map (for FunASR keys, language/ITN tokens, transducer geometry).</summary>
    public IReadOnlyDictionary<string, string> Metadata { get; init; } = new Dictionary<string, string>();
}
