using Stt.Core.Pipeline;

namespace Stt.Pipeline.Tests;

public class EndpointDetectorTests
{
    [Fact]
    public void Fires_On_Trailing_Silence()
    {
        var ep = new EndpointDetector(minTrailingSilenceSeconds: 1.0, maxUtteranceSeconds: 30);
        // 0.5s speech (no endpoint), then silence accumulating to 1.0s.
        Assert.False(ep.Update(isSpeech: true, 0.5));
        Assert.False(ep.Update(false, 0.5));
        Assert.False(ep.Update(false, 0.4)); // 0.9s silence
        Assert.True(ep.Update(false, 0.2));  // 1.1s silence ≥ 1.0
    }

    [Fact]
    public void Fires_On_Max_Utterance_Even_While_Speaking()
    {
        var ep = new EndpointDetector(1.0, maxUtteranceSeconds: 2.0);
        Assert.False(ep.Update(true, 1.0));
        Assert.True(ep.Update(true, 1.2)); // 2.2s utterance ≥ 2.0, despite continuous speech
    }

    [Fact]
    public void Leading_Silence_Does_Not_Trigger()
    {
        var ep = new EndpointDetector(0.5, 30);
        Assert.False(ep.Update(false, 1.0)); // silence before any speech
        Assert.False(ep.Update(false, 1.0));
        Assert.False(ep.Update(false, 1.0));
    }

    [Fact]
    public void Reset_Clears_Accumulators()
    {
        var ep = new EndpointDetector(0.5, 30);
        ep.Update(true, 0.3);
        Assert.True(ep.Update(false, 0.6));
        ep.Reset();
        Assert.False(ep.Update(false, 0.6)); // after reset, no speech seen yet → no trigger
    }
}
