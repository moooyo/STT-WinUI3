using Stt.Abstractions.Audio;

namespace Stt.Core.Vad;

/// <summary>
/// The Silero segmentation state machine, decoupled from the neural model so it can be unit
/// tested with scripted probabilities (spec §15). Given per-window speech probabilities and the
/// matching audio windows, it tracks onset/offset with hysteresis and emits a
/// <see cref="SpeechSegment"/> once trailing silence exceeds the configured minimum. Padding is
/// applied around the detected region; segments shorter than the minimum are dropped.
/// </summary>
public sealed class VadSegmenter
{
    private readonly VadOptions _o;
    private readonly Queue<SpeechSegment> _ready = new();

    private bool _triggered;
    private long _pos;                       // absolute samples consumed
    private long _segStartSample;
    private long _tempEndSample;             // 0 = no pending end
    private readonly List<float> _segBuf = new();
    private readonly Queue<float> _preRoll = new(); // lookback for pre-pad when not triggered

    public VadSegmenter(VadOptions options) => _o = options;

    public void AcceptWindow(float prob, ReadOnlySpan<float> window)
    {
        if (!_triggered)
        {
            if (prob >= _o.Threshold)
            {
                // Speech onset. Seed the segment with the pre-roll (pad preceding this window)
                // plus the current window itself, so no audio is lost when pad < window length.
                _segStartSample = _pos - _preRoll.Count;
                _segBuf.Clear();
                _segBuf.AddRange(_preRoll);
                for (int i = 0; i < window.Length; i++) _segBuf.Add(window[i]);
                _triggered = true;
                _tempEndSample = 0;
            }
            else
            {
                // Still silence: keep a rolling pre-roll of SpeechPadSamples for the next onset.
                foreach (float s in window)
                {
                    _preRoll.Enqueue(s);
                    while (_preRoll.Count > _o.SpeechPadSamples) _preRoll.Dequeue();
                }
            }
        }
        else
        {
            for (int i = 0; i < window.Length; i++) _segBuf.Add(window[i]);

            if (prob >= _o.Threshold)
            {
                _tempEndSample = 0;          // still speech → cancel any pending end
            }
            else if (prob < _o.NegThreshold)
            {
                long windowEnd = _pos + window.Length;
                if (_tempEndSample == 0) _tempEndSample = windowEnd;

                if (windowEnd - _tempEndSample >= _o.MinSilenceSamples)
                    CloseSegment(_tempEndSample + _o.SpeechPadSamples);
            }
        }

        _pos += window.Length;
    }

    /// <summary>Force-close any open segment at the current position (end of capture/utterance).</summary>
    public void Flush()
    {
        if (_triggered)
            CloseSegment(_pos);
    }

    public bool TryDequeue(out SpeechSegment seg)
    {
        if (_ready.Count > 0) { seg = _ready.Dequeue(); return true; }
        seg = null!;
        return false;
    }

    public void Reset()
    {
        _triggered = false;
        _pos = 0;
        _tempEndSample = 0;
        _segStartSample = 0;
        _segBuf.Clear();
        _preRoll.Clear();
        _ready.Clear();
    }

    private void CloseSegment(long endSampleAbs)
    {
        long lengthSamples = endSampleAbs - _segStartSample;
        if (lengthSamples >= _o.MinSpeechSamples && _segBuf.Count > 0)
        {
            int take = (int)Math.Min(lengthSamples, _segBuf.Count);
            var samples = _segBuf.GetRange(0, take).ToArray();
            _ready.Enqueue(new SpeechSegment(_segStartSample, samples));
        }

        _triggered = false;
        _tempEndSample = 0;
        _segBuf.Clear();
        _preRoll.Clear();
    }
}
