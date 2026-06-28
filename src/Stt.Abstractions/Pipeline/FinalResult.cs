namespace Stt.Abstractions.Pipeline;

/// <summary>
/// A committed result for a segment. In two-pass mode this replaces the segment's
/// <see cref="PartialResult"/> with the offline second pass's re-decoded text (spec §6).
/// </summary>
public sealed record FinalResult(int SegmentId, string Text);
