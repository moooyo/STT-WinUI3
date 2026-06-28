namespace Stt.Abstractions.Decoders;

/// <summary>
/// A decoder hypothesis. <see cref="IsFinal"/> distinguishes a streaming partial from a
/// committed result. <see cref="Timestamps"/> is per-token, in seconds (empty when a model
/// does not emit them).
/// </summary>
public sealed record AsrResult(
    string Text,
    IReadOnlyList<int> Tokens,
    IReadOnlyList<float> Timestamps,
    bool IsFinal)
{
    /// <summary>An empty, non-final result (decoder just reset / no audio yet).</summary>
    public static AsrResult Empty { get; } =
        new(string.Empty, Array.Empty<int>(), Array.Empty<float>(), IsFinal: false);
}
