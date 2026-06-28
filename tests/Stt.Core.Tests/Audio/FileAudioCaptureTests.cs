using Stt.Abstractions.Audio;
using Stt.Core.Audio;

namespace Stt.Core.Tests.Audio;

public class FileAudioCaptureTests
{
    [Fact]
    public async Task Streams_All_Samples_In_Frames()
    {
        string path = Path.Combine(Path.GetTempPath(), $"cap_{Guid.NewGuid():N}.wav");
        try
        {
            // 1 s @ 16k mono.
            var src = new float[16000];
            for (int i = 0; i < src.Length; i++) src[i] = MathF.Sin(2 * MathF.PI * 300f * i / 16000f) * 0.5f;
            WavIo.WritePcm16(path, src, 16000);

            var cap = new FileAudioCapture(path, frameSamples: 512, realTime: false);
            int total = 0, frames = 0;
            cap.FrameAvailable += f => { total += f.Count; frames++; };

            await cap.StartAsync(CancellationToken.None);

            Assert.Equal(src.Length, total);
            Assert.Equal((int)Math.Ceiling(src.Length / 512.0), frames);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public async Task Resamples_48k_Source_To_16k()
    {
        string path = Path.Combine(Path.GetTempPath(), $"cap48_{Guid.NewGuid():N}.wav");
        try
        {
            var src = new float[48000]; // 1 s @ 48k
            for (int i = 0; i < src.Length; i++) src[i] = MathF.Sin(2 * MathF.PI * 200f * i / 48000f) * 0.5f;
            WavIo.WritePcm16(path, src, 48000);

            var cap = new FileAudioCapture(path, frameSamples: 1600, realTime: false);
            int total = 0;
            cap.FrameAvailable += f => total += f.Count;
            await cap.StartAsync(CancellationToken.None);

            Assert.InRange(total, 15900, 16100); // ~16000 after 48k→16k
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public async Task Honors_Cancellation()
    {
        var cap = new FileAudioCapture(new float[16000], frameSamples: 512, realTime: true);
        using var cts = new CancellationTokenSource();
        cts.CancelAfter(20);
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => cap.StartAsync(cts.Token));
    }
}
