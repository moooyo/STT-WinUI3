using System.Threading.Channels;
using Stt.Abstractions.Audio;
using Stt.Abstractions.Common;
using Stt.Abstractions.Decoders;
using Stt.Abstractions.Features;
using Stt.Abstractions.Pipeline;
using Stt.Abstractions.Vad;
using Stt.Core.Audio;
using Stt.Core.Vad;

namespace Stt.Core.Pipeline;

/// <summary>
/// The speech-to-text pipeline (spec §5.1, §6). Phase 0 implements the <c>OnePassOffline</c> path:
/// audio frames are decoupled from inference by a bounded channel (backpressure #1: DropOldest,
/// never blocks the audio callback); a single long-running worker chunks frames into 512-sample
/// VAD windows, and for each emitted <see cref="SpeechSegment"/> runs the offline front-end +
/// decoder and raises <see cref="Final"/> through the <see cref="IUiDispatcher"/>. Streaming and
/// two-pass paths are added in Phase 1.
/// </summary>
public sealed class SttPipeline : ISttPipeline
{
    private const int VadWindow = 512;
    private const int AudioChannelCapacity = 64;

    private readonly IAudioCapture _capture;
    private readonly IVad _vad;
    private readonly IFeatureFrontend _offlineFrontend;
    private readonly IAsrDecoder _offlineDecoder;
    private readonly IUiDispatcher _ui;

    private Channel<AudioFrame>? _audio;
    private CancellationTokenSource? _cts;
    private Task? _worker;
    private Task? _captureTask;
    private int _segmentId;
    private long _droppedFrames;

    public SttPipeline(
        PipelineConfig config,
        IAudioCapture capture,
        IVad vad,
        IFeatureFrontend offlineFrontend,
        IAsrDecoder offlineDecoder,
        IUiDispatcher uiDispatcher)
    {
        Config = config;
        _capture = capture;
        _vad = vad;
        _offlineFrontend = offlineFrontend;
        _offlineDecoder = offlineDecoder;
        _ui = uiDispatcher;
    }

    public PipelineConfig Config { get; }

    /// <summary>Count of audio frames dropped by backpressure #1 (drives the UI "behind" indicator).</summary>
    public long DroppedFrames => Interlocked.Read(ref _droppedFrames);

    public event Action<PartialResult>? Partial;
    public event Action<FinalResult>? Final;

    public Task StartAsync(CancellationToken ct)
    {
        _cts = CancellationTokenSource.CreateLinkedTokenSource(ct);
        var token = _cts.Token;

        _audio = Channel.CreateBounded<AudioFrame>(new BoundedChannelOptions(AudioChannelCapacity)
        {
            FullMode = BoundedChannelFullMode.DropOldest,   // spec §6 backpressure #1: never block audio
            SingleReader = true,
            SingleWriter = true,
        });

        _capture.FrameAvailable += OnFrame;

        _worker = Task.Factory.StartNew(
            () => WorkerLoopAsync(token), token, TaskCreationOptions.LongRunning, TaskScheduler.Default)
            .Unwrap();

        _captureTask = Task.Run(async () =>
        {
            try { await _capture.StartAsync(token).ConfigureAwait(false); }
            catch (OperationCanceledException) { /* expected on stop */ }
            finally { _audio!.Writer.TryComplete(); }
        }, token);

        return Task.CompletedTask;
    }

    private void OnFrame(AudioFrame frame)
    {
        // Synchronous, non-blocking, non-allocating-ish: just enqueue (spec §6).
        if (_audio is null) return;
        if (!_audio.Writer.TryWrite(frame))
            Interlocked.Increment(ref _droppedFrames);
    }

    private async Task WorkerLoopAsync(CancellationToken ct)
    {
        var chunker = new FrameChunker(VadWindow, VadWindow);
        try
        {
            await foreach (AudioFrame frame in _audio!.Reader.ReadAllAsync(ct).ConfigureAwait(false))
            {
                chunker.Push(frame.Span);
                while (chunker.TryGetWindow(out float[] window))
                {
                    _vad.AcceptWaveform(window);
                    DrainSegments();
                }
            }

            // End of capture: flush the VAD so a segment without trailing silence still closes.
            FlushVad();
            DrainSegments();
        }
        catch (OperationCanceledException) { /* expected on stop */ }
    }

    private void DrainSegments()
    {
        while (_vad.TryDequeueSegment(out SpeechSegment seg))
            ProcessSegmentOffline(seg);
    }

    private void ProcessSegmentOffline(SpeechSegment seg)
    {
        int id = _segmentId++;

        float[] feats = _offlineFrontend.Extract(seg.Samples, out int numFrames);
        _offlineDecoder.Reset();
        _offlineDecoder.AcceptFeatures(feats, numFrames, _offlineFrontend.FeatureDim);
        _offlineDecoder.InputFinished();
        AsrResult result = _offlineDecoder.GetResult();

        _ui.Enqueue(() => Final?.Invoke(new FinalResult(id, result.Text)));
    }

    /// <summary>Force any open VAD segment to close at end of capture (feeds trailing silence; uses Flush when available).</summary>
    private void FlushVad()
    {
        if (_vad is SileroVad silero)
        {
            silero.Flush();
            return;
        }

        // Generic fallback: feed ~1.5 s of silence so a silence-based segmenter closes.
        int windows = 16000 * 3 / 2 / VadWindow + 2;
        var silence = new float[VadWindow];
        for (int i = 0; i < windows; i++)
        {
            _vad.AcceptWaveform(silence);
            DrainSegments();
        }
    }

    /// <summary>
    /// Await natural completion: the capture source ends (e.g. a file), the channel completes, and
    /// the worker drains all remaining segments. Used by headless file-based tests; unlike
    /// <see cref="StopAsync"/> it does not cancel, so in-flight segments are not dropped.
    /// </summary>
    internal async Task WaitForDrainAsync()
    {
        if (_captureTask is not null) await _captureTask.ConfigureAwait(false);
        if (_worker is not null) await _worker.ConfigureAwait(false);
    }

    public async Task StopAsync()
    {
        _cts?.Cancel();
        try
        {
            if (_captureTask is not null) await _captureTask.ConfigureAwait(false);
            if (_worker is not null) await _worker.ConfigureAwait(false);
        }
        catch (OperationCanceledException) { }
        finally
        {
            _capture.FrameAvailable -= OnFrame;
            await _capture.StopAsync().ConfigureAwait(false);
        }
    }

    public async ValueTask DisposeAsync()
    {
        await StopAsync().ConfigureAwait(false);
        _cts?.Dispose();
        _vad.Dispose();
        _offlineDecoder.Dispose();
        (_offlineFrontend as IDisposable)?.Dispose();
        _capture.Dispose();
    }
}
