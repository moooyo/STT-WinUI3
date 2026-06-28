using Stt.Core.Vad;

namespace Stt.Core.Tests.Vad;

/// <summary>
/// Integration test for the ONNX-backed <see cref="SileroVad"/>. Skips unless a model is
/// provided via the <c>STT_SILERO_VAD</c> environment variable (D7: models are user-supplied).
/// </summary>
public class SileroVadModelTests
{
    [SkippableFact]
    public void Detects_One_Segment_In_Tone_Burst()
    {
        string? modelPath = Environment.GetEnvironmentVariable("STT_SILERO_VAD");
        Skip.If(string.IsNullOrEmpty(modelPath) || !File.Exists(modelPath),
            "Set STT_SILERO_VAD to a silero_vad.onnx path to run this test.");

        var opts = new VadOptions { MinSilenceDurationMs = 300, MinSpeechDurationMs = 100, SpeechPadMs = 30 };
        using var vad = new SileroVad(modelPath!, opts);

        // 0.3s silence, 0.6s 300Hz tone, 1s silence — fed in 512-sample windows.
        var audio = new List<float>();
        audio.AddRange(new float[4800]);
        for (int i = 0; i < 9600; i++) audio.Add(MathF.Sin(2 * MathF.PI * 300f * i / 16000f) * 0.6f);
        audio.AddRange(new float[16000]);

        int segments = 0;
        for (int off = 0; off + 512 <= audio.Count; off += 512)
        {
            vad.AcceptWaveform(audio.GetRange(off, 512).ToArray());
            while (vad.TryDequeueSegment(out _)) segments++;
        }
        vad.Flush();
        while (vad.TryDequeueSegment(out _)) segments++;

        Assert.Equal(1, segments);
    }
}
