using Stt.Core.Audio;

namespace Stt.Core.Tests.Audio;

public class RingBufferTests
{
    [Fact]
    public void Fifo_Order_Preserved_Across_Wrap()
    {
        var rb = new RingBuffer(16);
        Assert.Equal(10, rb.Write(Enumerable.Range(0, 10).Select(i => (float)i).ToArray()));

        var first = new float[6];
        Assert.Equal(6, rb.Read(first));
        Assert.Equal(new float[] { 0, 1, 2, 3, 4, 5 }, first);

        // Write 8 more; head is at 6, tail at 10 → 4 left in front, wraps.
        Assert.Equal(8, rb.Write(Enumerable.Range(10, 8).Select(i => (float)i).ToArray()));

        var rest = new float[12];
        Assert.Equal(12, rb.Read(rest));
        Assert.Equal(new float[] { 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17 }, rest);
        Assert.Equal(0, rb.Count);
    }

    [Fact]
    public void Overflow_Truncates_And_Reports_Written()
    {
        var rb = new RingBuffer(4);
        int written = rb.Write(new float[] { 1, 2, 3, 4, 5, 6 });
        Assert.Equal(4, written);
        Assert.Equal(0, rb.Free);
    }

    [Fact]
    public void Read_From_Empty_Returns_Zero()
    {
        var rb = new RingBuffer(4);
        Assert.Equal(0, rb.Read(new float[4]));
    }
}
