using Stt.Core.Audio;

namespace Stt.Core.Tests.Audio;

public class WavIoTests
{
    [Fact]
    public void RoundTrip_Pcm16_Within_Quantization()
    {
        string path = Path.Combine(Path.GetTempPath(), $"wavio_{Guid.NewGuid():N}.wav");
        try
        {
            var src = new float[2000];
            for (int i = 0; i < src.Length; i++) src[i] = MathF.Sin(2 * MathF.PI * 440f * i / 16000f) * 0.8f;

            WavIo.WritePcm16(path, src, 16000);
            var read = WavIo.ReadPcm(path);

            Assert.Equal(16000, read.SampleRate);
            Assert.Equal(1, read.Channels);
            Assert.Equal(src.Length, read.Interleaved.Length);
            for (int i = 0; i < src.Length; i++)
                Assert.InRange(read.Interleaved[i] - src[i], -1f / 32768f, 1f / 32768f);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public void Rejects_NonRiff()
    {
        var garbage = new byte[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
        Assert.Throws<InvalidDataException>(() => WavIo.ParsePcm(garbage));
    }
}
