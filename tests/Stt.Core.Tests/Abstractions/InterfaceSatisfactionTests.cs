using Stt.Abstractions.Audio;
using Stt.Abstractions.Common;
using Stt.Abstractions.Decoders;
using Stt.Abstractions.Vad;

namespace Stt.Core.Tests.Abstractions;

/// <summary>
/// Compile-time proof that the Abstractions interfaces are implementable with only managed
/// types (no third-party deps leak into the contract). These fakes are reused by later tests.
/// </summary>
public class InterfaceSatisfactionTests
{
    private sealed class InlineUiDispatcher : IUiDispatcher
    {
        public int Count;
        public void Enqueue(Action action) { Count++; action(); }
    }

    private sealed class NoOpDecoder : IAsrDecoder
    {
        public DecoderCapabilities Capabilities => DecoderCapabilities.Offline;
        public void Reset() { }
        public bool AcceptFeatures(ReadOnlySpan<float> features, int numFrames, int featDim) => false;
        public void InputFinished() { }
        public bool IsEndpoint() => false;
        public AsrResult GetResult() => AsrResult.Empty;
        public void Dispose() { }
    }

    private sealed class NoOpVad : IVad
    {
        public void Reset() { }
        public void AcceptWaveform(ReadOnlySpan<float> window512) { }
        public bool TryDequeueSegment(out SpeechSegment seg) { seg = null!; return false; }
        public void Dispose() { }
    }

    [Fact]
    public void Fakes_SatisfyInterfaces()
    {
        IUiDispatcher ui = new InlineUiDispatcher();
        ui.Enqueue(() => { });
        Assert.Equal(1, ((InlineUiDispatcher)ui).Count);

        IAsrDecoder dec = new NoOpDecoder();
        Assert.False(dec.IsEndpoint());
        Assert.Equal(AsrResult.Empty, dec.GetResult());

        IVad vad = new NoOpVad();
        Assert.False(vad.TryDequeueSegment(out _));
    }
}
