namespace Stt.Abstractions.Audio;

/// <summary>
/// A speech region carved out by the VAD: a contiguous run of 16 kHz mono samples plus the
/// absolute sample offset where it began in the capture stream. Fed to the offline second
/// pass for re-decoding (spec §6).
/// </summary>
public sealed record SpeechSegment(long StartSample, float[] Samples)
{
    /// <summary>Duration of the segment in seconds at 16 kHz.</summary>
    public double DurationSeconds => Samples.Length / 16000.0;
}
