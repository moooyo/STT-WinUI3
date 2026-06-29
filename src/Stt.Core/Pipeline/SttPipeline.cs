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
/// The speech-to-text pipeline across all three modes (spec §6, §11): OnePassOffline,
/// OnePassStreaming, and TwoPass. Audio is decoupled from inference by a bounded DropOldest channel
/// (backpressure #1, never blocks the audio callback); a first-pass worker runs VAD + streaming
/// decode (live <see cref="Partial"/>) and pushes each VAD segment over a second bounded channel
/// using WriteAsync+Wait (backpressure #2, finals are precious); a second-pass worker re-decodes
/// offline and raises <see cref="Final"/>, replacing the partial by SegmentId. All UI updates cross
/// the boundary through the <see cref="IUiDispatcher"/>.
/// </summary>
public sealed class SttPipeline : ISttPipeline
{
    private const int VadWindow = 512;
    private const int AudioChannelCapacity = 64;
    private const int SegmentChannelCapacity = 8;

    private readonly IAudioCapture _capture;
    private readonly IVad _vad;
    private readonly IUiDispatcher _ui;
    private readonly IFeatureFrontend? _streamingFrontend;
    private readonly IAsrDecoder? _streamingDecoder;
    private readonly IFeatureFrontend? _offlineFrontend;
    private readonly IAsrDecoder? _offlineDecoder;

    private Channel<AudioFrame>? _audio;
    private Channel<(int Id, SpeechSegment Seg)>? _segments;
    private CancellationTokenSource? _cts;
    private Task? _firstPass;
    private Task? _secondPass;
    private Task? _captureTask;
    private int _segmentId;
    private long _droppedFrames;
    private long _lastPartialTick;

    public SttPipeline(
        PipelineConfig config,
        IAudioCapture capture,
        IVad vad,
        IUiDispatcher uiDispatcher,
        IFeatureFrontend? streamingFrontend = null,
        IAsrDecoder? streamingDecoder = null,
        IFeatureFrontend? offlineFrontend = null,
        IAsrDecoder? offlineDecoder = null)
    {
        Config = config;
        _capture = capture;
        _vad = vad;
        _ui = uiDispatcher;
        _streamingFrontend = streamingFrontend;
        _streamingDecoder = streamingDecoder;
        _offlineFrontend = offlineFrontend;
        _offlineDecoder = offlineDecoder;
        ValidateComponents();
    }

    /// <summary>Phase-0 compatible constructor for the OnePassOffline path.</summary>
    public SttPipeline(
        PipelineConfig config, IAudioCapture capture, IVad vad,
        IFeatureFrontend offlineFrontend, IAsrDecoder offlineDecoder, IUiDispatcher uiDispatcher)
        : this(config, capture, vad, uiDispatcher, offlineFrontend: offlineFrontend, offlineDecoder: offlineDecoder) { }

    public PipelineConfig Config { get; }
    public long DroppedFrames => Interlocked.Read(ref _droppedFrames);

    public event Action<PartialResult>? Partial;
    public event Action<FinalResult>? Final;

    /// <summary>Surfaces a worker-thread failure (missing shim, bad model, mic error) to the UI.</summary>
    public event Action<string>? Error;

    private void ValidateComponents()
    {
        switch (Config.Mode)
        {
            case PipelineMode.OnePassOffline:
                Require(_offlineFrontend is not null && _offlineDecoder is not null, "OnePassOffline requires an offline front-end + decoder.");
                break;
            case PipelineMode.OnePassStreaming:
                Require(_streamingFrontend is not null && _streamingDecoder is not null, "OnePassStreaming requires a streaming front-end + decoder.");
                break;
            case PipelineMode.TwoPass:
                Require(_streamingFrontend is not null && _streamingDecoder is not null, "TwoPass requires a streaming front-end + decoder.");
                Require(_offlineFrontend is not null && _offlineDecoder is not null, "TwoPass requires an offline front-end + decoder.");
                break;
        }
    }

    private static void Require(bool cond, string msg) { if (!cond) throw new ArgumentException(msg); }

    public Task StartAsync(CancellationToken ct)
    {
        _cts = CancellationTokenSource.CreateLinkedTokenSource(ct);
        var token = _cts.Token;

        _audio = Channel.CreateBounded<AudioFrame>(new BoundedChannelOptions(AudioChannelCapacity)
        {
            FullMode = BoundedChannelFullMode.DropOldest,   // backpressure #1: never block audio
            SingleReader = true,
            SingleWriter = true,
        }, itemDropped: _ => Interlocked.Increment(ref _droppedFrames));   // spec §6: itemDropped counter

        bool needsSecondPass = Config.Mode is PipelineMode.OnePassOffline or PipelineMode.TwoPass;
        if (needsSecondPass)
        {
            _segments = Channel.CreateBounded<(int, SpeechSegment)>(new BoundedChannelOptions(SegmentChannelCapacity)
            {
                FullMode = BoundedChannelFullMode.Wait,     // backpressure #2: finals are precious
                SingleReader = true,
                SingleWriter = true,
            });
            _secondPass = Task.Factory.StartNew(() => SecondPassLoopAsync(token), token,
                TaskCreationOptions.LongRunning, TaskScheduler.Default).Unwrap();
        }

        _capture.FrameAvailable += OnFrame;
        _firstPass = Task.Factory.StartNew(() => FirstPassLoopAsync(token), token,
            TaskCreationOptions.LongRunning, TaskScheduler.Default).Unwrap();

        _captureTask = Task.Run(async () =>
        {
            try { await _capture.StartAsync(token).ConfigureAwait(false); }
            catch (OperationCanceledException) { }
            finally { _audio!.Writer.TryComplete(); }
        }, token);

        return Task.CompletedTask;
    }

    private void OnFrame(AudioFrame frame) => TryIngest(frame);

    /// <summary>Initialize only the audio channel (no workers) so a test can exercise backpressure #1.</summary>
    internal void InitAudioChannelForTest()
    {
        _audio = Channel.CreateBounded<AudioFrame>(new BoundedChannelOptions(AudioChannelCapacity)
        {
            FullMode = BoundedChannelFullMode.DropOldest,
            SingleReader = true,
            SingleWriter = true,
        }, itemDropped: _ => Interlocked.Increment(ref _droppedFrames));
    }

    internal int AudioChannelCapacityForTest => AudioChannelCapacity;

    /// <summary>
    /// Enqueue a frame (backpressure #1). With DropOldest the write always succeeds; dropped frames
    /// are counted by the channel's itemDropped callback into <see cref="DroppedFrames"/>.
    /// </summary>
    internal bool TryIngest(AudioFrame frame) => _audio is not null && _audio.Writer.TryWrite(frame);

    private async Task FirstPassLoopAsync(CancellationToken ct)
    {
        var chunker = new FrameChunker(VadWindow, VadWindow);
        bool streaming = _streamingFrontend is not null && _streamingDecoder is not null;
        try
        {
            await foreach (AudioFrame frame in _audio!.Reader.ReadAllAsync(ct).ConfigureAwait(false))
            {
                if (streaming) RunStreaming(frame);

                chunker.Push(frame.Span);
                while (chunker.TryGetWindow(out float[] window))
                {
                    _vad.AcceptWaveform(window);
                    while (_vad.TryDequeueSegment(out SpeechSegment seg))
                        await OnSegmentAsync(seg, ct).ConfigureAwait(false);
                }
            }

            FlushVad();
            while (_vad.TryDequeueSegment(out SpeechSegment seg))
                await OnSegmentAsync(seg, ct).ConfigureAwait(false);
        }
        catch (OperationCanceledException) { }
        catch (Exception ex) { EmitError($"First pass failed: {ex.Message}"); }
        finally
        {
            _segments?.Writer.TryComplete();
        }
    }

    private void RunStreaming(AudioFrame frame)
    {
        float[] feats = _streamingFrontend!.Extract(frame.Span, out int nf);
        if (nf <= 0) return;
        if (_streamingDecoder!.AcceptFeatures(feats, nf, _streamingFrontend.FeatureDim))
            EmitPartial(_segmentId, _streamingDecoder.GetResult().Text);
    }

    private async Task OnSegmentAsync(SpeechSegment seg, CancellationToken ct)
    {
        int id = _segmentId;

        if (Config.Mode == PipelineMode.OnePassStreaming)
        {
            // Streaming-only: commit the streaming hypothesis as the final for this segment.
            _streamingDecoder!.InputFinished();
            EmitFinal(id, _streamingDecoder.GetResult().Text);
            _streamingDecoder.Reset();
        }
        else if (_segments is not null)
        {
            // Offline / TwoPass: hand the segment to the second pass (Wait = no drop).
            await _segments.Writer.WriteAsync((id, seg), ct).ConfigureAwait(false);
            _streamingDecoder?.Reset();
        }

        _segmentId++;
    }

    private async Task SecondPassLoopAsync(CancellationToken ct)
    {
        try
        {
            await foreach (var (id, seg) in _segments!.Reader.ReadAllAsync(ct).ConfigureAwait(false))
            {
                float[] feats = _offlineFrontend!.Extract(seg.Samples, out int nf);
                _offlineDecoder!.Reset();
                _offlineDecoder.AcceptFeatures(feats, nf, _offlineFrontend.FeatureDim);
                _offlineDecoder.InputFinished();
                EmitFinal(id, _offlineDecoder.GetResult().Text);
            }
        }
        catch (OperationCanceledException) { }
        catch (Exception ex) { EmitError($"Offline decode failed: {ex.Message}"); }
    }

    private void EmitPartial(int id, string text)
    {
        long now = Environment.TickCount64;
        if (Config.PartialCoalesceMilliseconds > 0 && now - _lastPartialTick < Config.PartialCoalesceMilliseconds)
            return;
        _lastPartialTick = now;
        _ui.Enqueue(() => Partial?.Invoke(new PartialResult(id, text)));
    }

    private void EmitFinal(int id, string text) =>
        _ui.Enqueue(() => Final?.Invoke(new FinalResult(id, text)));

    private void EmitError(string message) => _ui.Enqueue(() => Error?.Invoke(message));

    private void FlushVad()
    {
        if (_vad is SileroVad silero) { silero.Flush(); return; }
        int windows = 16000 * 3 / 2 / VadWindow + 2;
        var silence = new float[VadWindow];
        for (int i = 0; i < windows; i++) _vad.AcceptWaveform(silence);
    }

    internal async Task WaitForDrainAsync()
    {
        if (_captureTask is not null) await _captureTask.ConfigureAwait(false);
        if (_firstPass is not null) await _firstPass.ConfigureAwait(false);
        if (_secondPass is not null) await _secondPass.ConfigureAwait(false);
    }

    public async Task StopAsync()
    {
        _cts?.Cancel();
        try
        {
            if (_captureTask is not null) await _captureTask.ConfigureAwait(false);
            if (_firstPass is not null) await _firstPass.ConfigureAwait(false);
            if (_secondPass is not null) await _secondPass.ConfigureAwait(false);
        }
        catch { /* cancellation or a worker fault (e.g. missing native shim) — Stop must still complete */ }
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
        _streamingDecoder?.Dispose();
        _offlineDecoder?.Dispose();
        (_streamingFrontend as IDisposable)?.Dispose();
        (_offlineFrontend as IDisposable)?.Dispose();
        _capture.Dispose();
    }
}
