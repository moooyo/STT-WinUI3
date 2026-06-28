namespace Stt.Abstractions.Audio;

/// <summary>
/// A block of 16 kHz mono float samples. Backed by a pooled array (spec §6: the audio
/// callback thread never allocates), so <see cref="Count"/> may be less than
/// <c>Samples.Length</c> — only the first <see cref="Count"/> entries are valid.
/// </summary>
public sealed record AudioFrame(float[] Samples, int Count)
{
    /// <summary>The valid samples as a span.</summary>
    public ReadOnlySpan<float> Span => Samples.AsSpan(0, Count);
}
