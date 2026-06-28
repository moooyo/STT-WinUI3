using System.Collections.Concurrent;
using Stt.Abstractions.Audio;
using Stt.Abstractions.Common;
using Stt.Abstractions.Decoders;
using Stt.Abstractions.Features;
using Stt.Abstractions.Pipeline;
using Stt.Abstractions.Vad;
using Stt.Core.Audio;
using Stt.Core.Pipeline;

namespace Stt.Pipeline.Tests;

public class TwoPassPipelineTests
{
    private sealed class OneSegmentVad : IVad
    {
        private int _windows;
        private bool _emitted;
        private readonly Queue<SpeechSegment> _q = new();
        public void Reset() { }
        public void AcceptWaveform(ReadOnlySpan<float> w)
        {
            if (++_windows >= 5 && !_emitted) { _emitted = true; _q.Enqueue(new SpeechSegment(0, new float[1600])); }
        }
        public bool TryDequeueSegment(out SpeechSegment seg)
        {
            if (_q.Count > 0) { seg = _q.Dequeue(); return true; }
            seg = null!; return false;
        }
        public void Dispose() { }
    }

    private sealed class OneFramePerCallFrontend : IFeatureFrontend
    {
        public AsrFeatureFamily Family => AsrFeatureFamily.KaldiFbankPovey;
        public int FeatureDim => 80;
        public float[] Extract(ReadOnlySpan<float> pcm, out int n) { n = 1; return new float[80]; }
    }

    // Emits partials 你 → 你好 on the first two accepts, then stops advancing.
    private sealed class ScriptedStreamingDecoder : IAsrDecoder
    {
        private int _calls;
        private string _text = string.Empty;
        public DecoderCapabilities Capabilities => DecoderCapabilities.Streaming | DecoderCapabilities.PartialResults;
        public void Reset() { _calls = 0; _text = string.Empty; }
        public bool AcceptFeatures(ReadOnlySpan<float> f, int n, int d)
        {
            _calls++;
            if (_calls == 1) { _text = "你"; return true; }
            if (_calls == 2) { _text = "你好"; return true; }
            return false;
        }
        public void InputFinished() { }
        public bool IsEndpoint() => false;
        public AsrResult GetResult() => new(_text, Array.Empty<int>(), Array.Empty<float>(), IsFinal: false);
        public void Dispose() { }
    }

    private sealed class FixedOfflineDecoder : IAsrDecoder
    {
        private AsrResult _r = AsrResult.Empty;
        public DecoderCapabilities Capabilities => DecoderCapabilities.Offline;
        public void Reset() => _r = AsrResult.Empty;
        public bool AcceptFeatures(ReadOnlySpan<float> f, int n, int d) => true;
        public void InputFinished() => _r = new AsrResult("你好 world", Array.Empty<int>(), Array.Empty<float>(), true);
        public bool IsEndpoint() => false;
        public AsrResult GetResult() => _r;
        public void Dispose() { }
    }

    private sealed class InlineDispatcher : IUiDispatcher { public void Enqueue(Action a) => a(); }

    [Fact]
    public async Task TwoPass_Emits_Partials_Then_One_Final_Replacing_Them()
    {
        string path = Path.Combine(Path.GetTempPath(), $"twopass_{Guid.NewGuid():N}.wav");
        var src = new float[16000];
        for (int i = 0; i < src.Length; i++) src[i] = MathF.Sin(2 * MathF.PI * 250f * i / 16000f) * 0.4f;
        WavIo.WritePcm16(path, src, 16000);

        try
        {
            var partials = new ConcurrentQueue<PartialResult>();
            var finals = new ConcurrentQueue<FinalResult>();

            var pipeline = new SttPipeline(
                new PipelineConfig { Mode = PipelineMode.TwoPass, PartialCoalesceMilliseconds = 0 },
                new FileAudioCapture(path, frameSamples: 512, realTime: false),
                new OneSegmentVad(),
                new InlineDispatcher(),
                streamingFrontend: new OneFramePerCallFrontend(),
                streamingDecoder: new ScriptedStreamingDecoder(),
                offlineFrontend: new OneFramePerCallFrontend(),
                offlineDecoder: new FixedOfflineDecoder());

            pipeline.Partial += partials.Enqueue;
            pipeline.Final += finals.Enqueue;

            await pipeline.StartAsync(CancellationToken.None);
            await pipeline.WaitForDrainAsync();
            await pipeline.DisposeAsync();

            // The first segment's partials progressed 你 → 你好.
            var seg0Partials = partials.Where(p => p.SegmentId == 0).Select(p => p.Text).ToList();
            Assert.Contains("你", seg0Partials);
            Assert.Contains("你好", seg0Partials);

            // Exactly one final for segment 0, replacing the partial with the re-decoded text.
            var seg0Finals = finals.Where(f => f.SegmentId == 0).ToList();
            Assert.Single(seg0Finals);
            Assert.Equal("你好 world", seg0Finals[0].Text);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }
}
