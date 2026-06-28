using Stt.Abstractions.Audio;
using Stt.Abstractions.Common;
using Stt.Abstractions.Decoders;
using Stt.Abstractions.Features;
using Stt.Abstractions.Pipeline;
using Stt.Abstractions.Vad;
using Stt.Core.Audio;
using Stt.Core.Pipeline;

namespace Stt.Pipeline.Tests;

public class OfflinePipelineTests
{
    // Emits exactly one segment after a few windows, regardless of content.
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

    private sealed class FakeFrontend : IFeatureFrontend
    {
        public AsrFeatureFamily Family => AsrFeatureFamily.KaldiFbankLfrCmvn;
        public int FeatureDim => 560;
        public float[] Extract(ReadOnlySpan<float> pcm, out int numFrames) { numFrames = 4; return new float[4 * 560]; }
    }

    private sealed class FixedTextDecoder : IAsrDecoder
    {
        private readonly string _text;
        private AsrResult _r = AsrResult.Empty;
        public FixedTextDecoder(string text) => _text = text;
        public DecoderCapabilities Capabilities => DecoderCapabilities.Offline;
        public void Reset() => _r = AsrResult.Empty;
        public bool AcceptFeatures(ReadOnlySpan<float> f, int n, int d) => true;
        public void InputFinished() => _r = new AsrResult(_text, Array.Empty<int>(), Array.Empty<float>(), true);
        public bool IsEndpoint() => false;
        public AsrResult GetResult() => _r;
        public void Dispose() { }
    }

    private sealed class InlineDispatcher : IUiDispatcher
    {
        public void Enqueue(Action action) => action();
    }

    [Fact]
    public async Task OnePassOffline_Emits_One_Final_With_Decoder_Text()
    {
        // 1 s of 16k audio from a generated WAV → file capture → pipeline.
        string path = Path.Combine(Path.GetTempPath(), $"pipe_{Guid.NewGuid():N}.wav");
        var src = new float[16000];
        for (int i = 0; i < src.Length; i++) src[i] = MathF.Sin(2 * MathF.PI * 250f * i / 16000f) * 0.4f;
        WavIo.WritePcm16(path, src, 16000);

        try
        {
            var finals = new List<FinalResult>();
            var pipeline = new SttPipeline(
                new PipelineConfig { Mode = PipelineMode.OnePassOffline },
                new FileAudioCapture(path, frameSamples: 512, realTime: false),
                new OneSegmentVad(),
                new FakeFrontend(),
                new FixedTextDecoder("你好 world"),
                new InlineDispatcher());

            pipeline.Final += f => finals.Add(f);

            await pipeline.StartAsync(CancellationToken.None);
            await pipeline.WaitForDrainAsync();
            await pipeline.DisposeAsync();

            Assert.Single(finals);
            Assert.Equal("你好 world", finals[0].Text);
            Assert.Equal(0, finals[0].SegmentId);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }
}
