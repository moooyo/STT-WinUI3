namespace Stt.Abstractions.Pipeline;

/// <summary>A live, in-progress hypothesis for a segment (rendered grey/italic in the UI).</summary>
public sealed record PartialResult(int SegmentId, string Text);
