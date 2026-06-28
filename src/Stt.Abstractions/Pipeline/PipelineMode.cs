namespace Stt.Abstractions.Pipeline;

/// <summary>Pipeline operating mode (spec §11).</summary>
public enum PipelineMode
{
    /// <summary>Streaming partials only, no final re-decode.</summary>
    OnePassStreaming,

    /// <summary>VAD segments → offline whole-segment transcription (no per-word, sentence latency).</summary>
    OnePassOffline,

    /// <summary>Streaming partials + end-of-sentence offline re-decode replacing the partial (default).</summary>
    TwoPass
}
