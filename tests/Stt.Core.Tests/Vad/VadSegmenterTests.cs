using Stt.Core.Vad;

namespace Stt.Core.Tests.Vad;

public class VadSegmenterTests
{
    private static readonly VadOptions Opts = new()
    {
        SampleRate = 16000,
        WindowSamples = 512,
        Threshold = 0.5f,
        NegThreshold = 0.35f,
        MinSilenceDurationMs = 200,   // ~6 windows of 512 (32ms each)
        MinSpeechDurationMs = 100,    // ~3 windows
        SpeechPadMs = 0,              // no padding for exact assertions
    };

    private static float[] Window(float v = 0f)
    {
        var w = new float[512];
        Array.Fill(w, v);
        return w;
    }

    [Fact]
    public void Emits_One_Segment_For_Speech_Then_Silence()
    {
        var seg = new VadSegmenter(Opts);

        // 4 windows silence, 20 windows speech (~640ms), then 12 windows silence (~384ms > 200ms).
        for (int i = 0; i < 4; i++) seg.AcceptWindow(0.05f, Window(0.01f));
        for (int i = 0; i < 20; i++) seg.AcceptWindow(0.95f, Window(0.5f));
        for (int i = 0; i < 12; i++) seg.AcceptWindow(0.02f, Window(0.0f));

        Assert.True(seg.TryDequeue(out var s));
        Assert.False(seg.TryDequeue(out _)); // exactly one
        // Segment should roughly span the 20 speech windows (~10240 samples), within silence slack.
        Assert.InRange(s.Samples.Length, 20 * 512, 27 * 512);
    }

    [Fact]
    public void Drops_Sub_Minimum_Blip()
    {
        var seg = new VadSegmenter(Opts);
        for (int i = 0; i < 4; i++) seg.AcceptWindow(0.05f, Window());
        // Only 1 window of speech (~32ms < 100ms min) then silence.
        seg.AcceptWindow(0.95f, Window(0.5f));
        for (int i = 0; i < 12; i++) seg.AcceptWindow(0.02f, Window());

        Assert.False(seg.TryDequeue(out _));
    }

    [Fact]
    public void Hysteresis_Keeps_Segment_Through_Brief_Dip()
    {
        var seg = new VadSegmenter(Opts);
        for (int i = 0; i < 10; i++) seg.AcceptWindow(0.9f, Window(0.5f));
        // Dip to 0.4 (between neg=0.35 and thr=0.5) — should NOT end speech.
        for (int i = 0; i < 5; i++) seg.AcceptWindow(0.4f, Window(0.3f));
        for (int i = 0; i < 10; i++) seg.AcceptWindow(0.9f, Window(0.5f));
        for (int i = 0; i < 12; i++) seg.AcceptWindow(0.02f, Window());

        Assert.True(seg.TryDequeue(out var s));
        // The dip stayed inside one segment → length covers all ~25 speech-ish windows.
        Assert.InRange(s.Samples.Length, 25 * 512, 33 * 512);
    }

    [Fact]
    public void Flush_Closes_Open_Segment_At_End_Of_Capture()
    {
        var seg = new VadSegmenter(Opts);
        for (int i = 0; i < 4; i++) seg.AcceptWindow(0.05f, Window());
        for (int i = 0; i < 20; i++) seg.AcceptWindow(0.95f, Window(0.5f));
        // No trailing silence — but capture ended.
        Assert.False(seg.TryDequeue(out _));
        seg.Flush();
        Assert.True(seg.TryDequeue(out var s));
        Assert.InRange(s.Samples.Length, 20 * 512, 21 * 512);
    }
}
