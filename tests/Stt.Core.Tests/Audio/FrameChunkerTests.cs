using Stt.Core.Audio;

namespace Stt.Core.Tests.Audio;

public class FrameChunkerTests
{
    [Fact]
    public void Window512_Hop512_Yields_Two_Windows_From_1200()
    {
        var fc = new FrameChunker(512, 512);
        fc.Push(new float[1200]);

        Assert.True(fc.TryGetWindow(out var w1));
        Assert.Equal(512, w1.Length);
        Assert.True(fc.TryGetWindow(out _));
        Assert.False(fc.TryGetWindow(out _));   // 1200 - 1024 = 176 left
        Assert.Equal(176, fc.Buffered);
    }

    [Fact]
    public void Overlap_Window400_Hop160_Retains_Tail()
    {
        var fc = new FrameChunker(400, 160);
        // ramp so we can verify overlap content
        var ramp = Enumerable.Range(0, 800).Select(i => (float)i).ToArray();
        fc.Push(ramp);

        Assert.True(fc.TryGetWindow(out var w1));
        Assert.Equal(0f, w1[0]);
        Assert.Equal(399f, w1[399]);

        Assert.True(fc.TryGetWindow(out var w2));
        // second window starts hop=160 later
        Assert.Equal(160f, w2[0]);
        Assert.Equal(559f, w2[399]);
    }

    [Fact]
    public void No_Window_When_Underfilled()
    {
        var fc = new FrameChunker(512, 512);
        fc.Push(new float[100]);
        Assert.False(fc.TryGetWindow(out _));
        Assert.Equal(100, fc.Buffered);
    }
}
