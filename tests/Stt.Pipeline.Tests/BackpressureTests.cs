using System.Collections.Concurrent;
using Stt.Abstractions.Audio;
using Stt.Abstractions.Common;
using Stt.Abstractions.Decoders;
using Stt.Abstractions.Features;
using Stt.Abstractions.Pipeline;
using Stt.Abstractions.Vad;
using Stt.Core.Pipeline;

namespace Stt.Pipeline.Tests;

public class BackpressureTests
{
    private sealed class NoopVad : IVad
    {
        public void Reset() { }
        public void AcceptWaveform(ReadOnlySpan<float> w) { }
        public bool TryDequeueSegment(out SpeechSegment seg) { seg = null!; return false; }
        public void Dispose() { }
    }
    private sealed class NoopFrontend : IFeatureFrontend
    {
        public AsrFeatureFamily Family => AsrFeatureFamily.KaldiFbankLfrCmvn;
        public int FeatureDim => 560;
        public float[] Extract(ReadOnlySpan<float> pcm, out int n) { n = 0; return Array.Empty<float>(); }
    }
    private sealed class NoopDecoder : IAsrDecoder
    {
        public DecoderCapabilities Capabilities => DecoderCapabilities.Offline;
        public void Reset() { }
        public bool AcceptFeatures(ReadOnlySpan<float> f, int n, int d) => false;
        public void InputFinished() { }
        public bool IsEndpoint() => false;
        public AsrResult GetResult() => AsrResult.Empty;
        public void Dispose() { }
    }
    private sealed class NoopCapture : IAudioCapture
    {
        public event Action<AudioFrame>? FrameAvailable;
        public Task StartAsync(CancellationToken ct) => Task.CompletedTask;
        public Task StopAsync() => Task.CompletedTask;
        public void Dispose() { _ = FrameAvailable; }
    }
    private sealed class InlineDispatcher : IUiDispatcher { public void Enqueue(Action a) => a(); }

    [Fact]
    public void Backpressure1_DropOldest_Counts_Drops_And_Never_Blocks()
    {
        var pipeline = new SttPipeline(
            new PipelineConfig { Mode = PipelineMode.OnePassOffline },
            new NoopCapture(), new NoopVad(), new NoopFrontend(), new NoopDecoder(), new InlineDispatcher());

        pipeline.InitAudioChannelForTest();   // channel only, no consumer
        int cap = pipeline.AudioChannelCapacityForTest;

        // Write cap + 10 frames with no consumer; DropOldest keeps the newest, drops 10.
        for (int i = 0; i < cap + 10; i++)
        {
            bool ok = pipeline.TryIngest(new AudioFrame(new float[1], 1));
            Assert.True(ok); // DropOldest write never blocks/fails
        }

        Assert.Equal(10, pipeline.DroppedFrames);
    }
}
