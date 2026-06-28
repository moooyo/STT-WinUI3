namespace Stt.Abstractions.Pipeline;

/// <summary>
/// The end-to-end speech-to-text pipeline (spec §5.1). Raises <see cref="Partial"/> for live
/// hypotheses (grey) and <see cref="Final"/> for committed text (black, replacing the partial).
/// All events are already marshalled to the UI thread via the engine's <c>IUiDispatcher</c>.
/// </summary>
public interface ISttPipeline : IAsyncDisposable
{
    /// <summary>The configuration this pipeline was built with.</summary>
    PipelineConfig Config { get; }

    /// <summary>Live partial hypothesis for a segment.</summary>
    event Action<PartialResult> Partial;

    /// <summary>Committed final result for a segment.</summary>
    event Action<FinalResult> Final;

    /// <summary>Start capture + inference workers.</summary>
    Task StartAsync(CancellationToken ct);

    /// <summary>Stop and drain workers (finals already queued are still delivered).</summary>
    Task StopAsync();
}
