using Stt.Abstractions.Common;
using Stt.Abstractions.Ep;
using Stt.Abstractions.Models;
using Stt.Abstractions.Pipeline;
using Stt.Audio.Windows;
using Stt.Core.Ep;
using Stt.Core.Pipeline;

namespace Stt.App.Services;

/// <summary>
/// App-facing transcription orchestrator (spec §12). Composes and runs the pipeline for the
/// selected mode — OnePassOffline, OnePassStreaming, or TwoPass — surfacing
/// <see cref="Partial"/>/<see cref="Final"/> events (already marshalled to the UI thread) and a
/// "behind" indicator.
/// </summary>
public sealed class TranscriptionService
{
    private readonly IModelRegistry _registry;
    private readonly IExecutionProviderSelector _epSelector;
    private readonly IUiDispatcher _ui;

    private SttPipeline? _pipeline;
    private OfflineChain? _offline;
    private StreamingChain? _streaming;
    private CancellationTokenSource? _cts;

    public TranscriptionService(IModelRegistry registry, IExecutionProviderSelector epSelector, IUiDispatcher ui)
    {
        _registry = registry;
        _epSelector = epSelector;
        _ui = ui;
    }

    public bool IsRunning { get; private set; }
    public long DroppedFrames => _pipeline?.DroppedFrames ?? 0;

    /// <summary>The EP actually bound after build (may differ from request if the EP fell back to CPU).</summary>
    public string ActiveProvider
    {
        get
        {
            string ep = _epSelector.LastResolution is { } r && !r.FellBackToCpu ? r.Device.Kind.ToString() : "CPU";
            return Stt.Core.Ep.EpDiagnostics.LastFallbackReason is { } why ? $"{ep} — {why}" : ep;
        }
    }

    public event Action<PartialResult>? Partial;
    public event Action<FinalResult>? Final;
    public event Action<string>? Error;

    public async Task StartAsync(SttOptions options, string? captureDeviceId = null)
    {
        if (IsRunning) return;

        if (string.IsNullOrEmpty(options.VadModelPath) || !File.Exists(options.VadModelPath))
            throw new InvalidOperationException("A Silero VAD model (.onnx) is required. Set it in Settings.");

        var vad = new Stt.Core.Vad.SileroVad(options.VadModelPath, new Stt.Core.Vad.VadOptions
        {
            MinSilenceDurationMs = (int)(options.MinTrailingSilenceSeconds * 1000),
        });
        var capture = new WasapiAudioCapture(WasapiAudioCapture.GetDeviceById(captureDeviceId));
        var pref = new EpPreference(options.Ep);
        Stt.Core.Ep.EpDiagnostics.LastFallbackReason = null;

        bool needStreaming = options.Mode is PipelineMode.OnePassStreaming or PipelineMode.TwoPass;
        bool needOffline = options.Mode is PipelineMode.OnePassOffline or PipelineMode.TwoPass;

        // Build sessions off the UI thread — loading ONNX + first-time TensorRT/DirectML engine
        // compile can take seconds–minutes and would otherwise freeze the window.
        await Task.Run(() =>
        {
            if (needStreaming)
            {
                if (string.IsNullOrEmpty(options.FirstPassModelId))
                    throw new InvalidOperationException("Select a streaming first-pass model (Settings).");
                var manifest = _registry.Get(options.FirstPassModelId);
                _streaming = StreamingPipelineBuilder.BuildStreaming(manifest, _epSelector, pref, options.MinTrailingSilenceSeconds);
            }

            if (needOffline)
            {
                if (string.IsNullOrEmpty(options.SecondPassModelId))
                    throw new InvalidOperationException("Select an offline model (Settings).");
                var manifest = _registry.Get(options.SecondPassModelId);
                _offline = OfflinePipelineBuilder.BuildOffline(manifest, options.VadModelPath, _epSelector, pref);
            }
        }).ConfigureAwait(false);

        _pipeline = new SttPipeline(
            options.ToPipelineConfig(), capture, vad, _ui,
            streamingFrontend: _streaming?.Frontend,
            streamingDecoder: _streaming?.Decoder,
            offlineFrontend: _offline?.Frontend,
            offlineDecoder: _offline?.Decoder);

        _pipeline.Partial += OnPartial;
        _pipeline.Final += OnFinal;
        _pipeline.Error += OnError;

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
        _pipeline.Error -= OnError;
        await _pipeline.DisposeAsync().ConfigureAwait(false);
        // The pipeline disposes the front-ends/decoders it owns; the chain records are now spent.
        _pipeline = null;
        _offline = null;
        _streaming = null;
        IsRunning = false;
    }

    private void OnPartial(PartialResult p) => Partial?.Invoke(p);
    private void OnFinal(FinalResult f) => Final?.Invoke(f);
    private void OnError(string m) => Error?.Invoke(m);
}
