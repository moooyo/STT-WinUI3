namespace Stt.Core.Vad;

/// <summary>
/// Silero VAD tuning (spec §11: endpoint silence 0.5–1.2 s). Durations are converted to sample
/// counts at <see cref="SampleRate"/>. <see cref="NegThreshold"/> gives hysteresis so brief dips
/// below <see cref="Threshold"/> don't prematurely end speech.
/// </summary>
public sealed record VadOptions
{
    public int SampleRate { get; init; } = 16000;
    public int WindowSamples { get; init; } = 512;

    /// <summary>Speech onset probability threshold.</summary>
    public float Threshold { get; init; } = 0.5f;

    /// <summary>Speech offset threshold (hysteresis). Defaults to Threshold - 0.15.</summary>
    public float NegThreshold { get; init; } = 0.35f;

    /// <summary>Trailing silence that closes a segment.</summary>
    public int MinSilenceDurationMs { get; init; } = 600;

    /// <summary>Minimum speech length to emit (drops blips).</summary>
    public int MinSpeechDurationMs { get; init; } = 250;

    /// <summary>Padding added before/after the detected speech region.</summary>
    public int SpeechPadMs { get; init; } = 90;

    public int MinSilenceSamples => MinSilenceDurationMs * SampleRate / 1000;
    public int MinSpeechSamples => MinSpeechDurationMs * SampleRate / 1000;
    public int SpeechPadSamples => SpeechPadMs * SampleRate / 1000;
}
