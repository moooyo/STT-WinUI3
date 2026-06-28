using Stt.Abstractions.Decoders;
using Stt.Abstractions.Ep;
using Stt.Abstractions.Pipeline;

namespace Stt.Core.Tests.Abstractions;

public class DtoTests
{
    [Fact]
    public void Records_HaveValueEquality()
    {
        var a = new PartialResult(1, "hi");
        var b = new PartialResult(1, "hi");
        Assert.Equal(a, b);

        var f1 = new FinalResult(2, "done");
        var f2 = new FinalResult(2, "done");
        Assert.Equal(f1, f2);
        Assert.NotEqual(f1, new FinalResult(3, "done"));
    }

    [Fact]
    public void DecoderCapabilities_FlagsCombine()
    {
        var caps = DecoderCapabilities.Streaming | DecoderCapabilities.Offline;
        Assert.True(caps.HasFlag(DecoderCapabilities.Streaming));
        Assert.True(caps.HasFlag(DecoderCapabilities.Offline));
        Assert.False(caps.HasFlag(DecoderCapabilities.Timestamps));
    }

    [Fact]
    public void EpPreference_DefaultsToCpuFallback()
    {
        var p = new EpPreference(EpKind.DirectML);
        Assert.True(p.AllowFallbackToCpu);
        Assert.Equal(EpKind.DirectML, p.Kind);
    }

    [Fact]
    public void PipelineConfig_DefaultsToTwoPass()
    {
        var c = new PipelineConfig();
        Assert.Equal(PipelineMode.TwoPass, c.Mode);
        Assert.True(c.MinTrailingSilenceSeconds is > 0.4f and < 1.3f);
    }
}
