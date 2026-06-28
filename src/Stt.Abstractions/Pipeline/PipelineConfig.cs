using Stt.Abstractions.Ep;

namespace Stt.Abstractions.Pipeline;

/// <summary>
/// Runtime configuration for a pipeline instance (spec §11). Model ids reference entries in the
/// <c>IModelRegistry</c>. Endpoint thresholds drive the VAD-silence / max-duration end-of-segment
/// rule (spec §11: silence 0.5–1.2 s OR a max-duration cap).
/// </summary>
public sealed record PipelineConfig
{
    /// <summary>Pipeline mode. Default is two-pass (spec §11 default).</summary>
    public PipelineMode Mode { get; init; } = PipelineMode.TwoPass;

    /// <summary>Streaming first-pass model id (must be streaming-capable). Null in OnePassOffline.</summary>
    public string? FirstPassModelId { get; init; }

    /// <summary>Offline second-pass model id (must be offline-capable). Used in TwoPass/OnePassOffline.</summary>
    public string? SecondPassModelId { get; init; }

    /// <summary>Execution provider preference.</summary>
    public EpPreference Ep { get; init; } = EpPreference.Auto;

    /// <summary>Trailing-silence seconds that ends a segment (spec §11: 0.5–1.2 s).</summary>
    public float MinTrailingSilenceSeconds { get; init; } = 0.8f;

    /// <summary>Hard cap on utterance length so a non-stop speaker still gets segmented.</summary>
    public float MaxUtteranceSeconds { get; init; } = 20f;

    /// <summary>How often (ms) streaming partials are coalesced before marshalling to the UI (spec §6).</summary>
    public int PartialCoalesceMilliseconds { get; init; } = 120;
}
