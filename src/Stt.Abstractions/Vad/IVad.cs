using Stt.Abstractions.Audio;

namespace Stt.Abstractions.Vad;

/// <summary>
/// Voice activity detector (spec §5.1). Fed fixed 512-sample windows at 16 kHz; emits a
/// <see cref="SpeechSegment"/> once a speech region is bounded by trailing silence.
/// </summary>
public interface IVad : IDisposable
{
    /// <summary>Clear internal state (hidden RNN state + accumulation) between utterances/sessions.</summary>
    void Reset();

    /// <summary>Push one 512-sample window (16 kHz mono).</summary>
    void AcceptWaveform(ReadOnlySpan<float> window512);

    /// <summary>Dequeue a completed speech segment if one is ready.</summary>
    bool TryDequeueSegment(out SpeechSegment seg);
}
