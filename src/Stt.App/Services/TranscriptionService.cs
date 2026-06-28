using Stt.Abstractions.Common;
using Stt.Abstractions.Ep;
using Stt.Abstractions.Models;
using Stt.Abstractions.Pipeline;
using Stt.Audio.Windows;
using Stt.Core.Ep;
using Stt.Core.Pipeline;

namespace Stt.App.Services;

/// <summary>
/// App-facing transcription orchestrator (spec §12). Builds and runs the OnePassOffline pipeline
/// (Phase 0) from the selected models, surfacing <see cref="Partial"/>/<see cref="Final"/> events
/// (already marshalled to the UI thread) and a "behind" indicator. Streaming/two-pass is added in
/// Phase 1.
/// </summary>
public sealed class TranscriptionService
{
    private readonly IModelRegistry _registry;
    private readonly IExecutionProviderSelector _epSelector;
    private readonly IUiDispatcher _ui;

    private SttPipeline? _pipeline;
    private OfflineChain? _chain;
    private CancellationTokenSource? _cts;

    public TranscriptionService(IModelRegistry registry, IExecutionProviderSelector epSelector, IUiDispatcher ui)
    {
        _registry = registry;
        _epSelector = epSelector;
        _ui = ui;
    }

    public bool IsRunning { get; private set; }
    public long DroppedFrames => _pipeline?.DroppedFrames ?? 0;

    public event Action<PartialResult>? Partial;
    public event Action<FinalResult>? Final;

    /// <summary>Start OnePassOffline transcription using <paramref name="options"/>.</summary>
    public async Task StartAsync(SttOptions options)
    {
        if (IsRunning) return;

        if (string.IsNullOrEmpty(options.SecondPassModelId))
            throw new InvalidOperationException("Select an offline model first (Model Manager).");
        if (string.IsNullOrEmpty(options.VadModelPath) || !File.Exists(options.VadModelPath))
            throw new InvalidOperationException("A Silero VAD model (.onnx) is required. Set it in Settings.");

        ModelManifest manifest = _registry.Get(options.SecondPassModelId);
        _chain = OfflinePipelineBuilder.BuildOffline(manifest, options.VadModelPath, _epSelector, new EpPreference(options.Ep));

        var capture = new WasapiAudioCapture();
        _pipeline = new SttPipeline(
            options.ToPipelineConfig(), capture, _chain.Vad, _chain.Frontend, _chain.Decoder, _ui);
        _pipeline.Partial += OnPartial;
        _pipeline.Final += OnFinal;

        _cts = new CancellationTokenSource();
        await _pipeline.StartAsync(_cts.Token).ConfigureAwait(false);
        IsRunning = true;
    }

    public async Task StopAsync()
    {
        if (!IsRunning || _pipeline is null) return;
        _cts?.Cancel();
        await _pipeline.StopAsync().ConfigureAwait(false);
        _pipeline.Partial -= OnPartial;
        _pipeline.Final -= OnFinal;
        await _pipeline.DisposeAsync().ConfigureAwait(false);
        _chain?.Dispose();
        _pipeline = null;
        _chain = null;
        IsRunning = false;
    }

    private void OnPartial(PartialResult p) => Partial?.Invoke(p);
    private void OnFinal(FinalResult f) => Final?.Invoke(f);
}
