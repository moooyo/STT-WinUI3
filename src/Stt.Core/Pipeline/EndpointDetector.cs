namespace Stt.Core.Pipeline;

/// <summary>
/// Decides when a streaming utterance has ended (spec §11): trailing VAD silence ≥
/// <c>minTrailingSilence</c> <b>or</b> the utterance has run past <c>maxUtterance</c> (so a
/// non-stop speaker is still segmented). Leading silence before any speech never triggers.
/// Pure and frame-driven so it is unit testable; the streaming first pass calls
/// <see cref="Update"/> per chunk and resets after an endpoint.
/// </summary>
public sealed class EndpointDetector
{
    private readonly double _minTrailingSilence;
    private readonly double _maxUtterance;

    private double _silenceAccum;
    private double _utteranceAccum;
    private bool _sawSpeech;

    public EndpointDetector(double minTrailingSilenceSeconds, double maxUtteranceSeconds)
    {
        _minTrailingSilence = minTrailingSilenceSeconds;
        _maxUtterance = maxUtteranceSeconds;
    }

    /// <summary>
    /// Advance by a chunk of <paramref name="elapsedSeconds"/> labelled speech or silence.
    /// Returns true when the endpoint rule fires (the caller should then flush and
    /// <see cref="Reset"/>).
    /// </summary>
    public bool Update(bool isSpeech, double elapsedSeconds)
    {
        _utteranceAccum += elapsedSeconds;
        if (isSpeech)
        {
            _sawSpeech = true;
            _silenceAccum = 0;
        }
        else
        {
            _silenceAccum += elapsedSeconds;
        }

        if (_sawSpeech && _silenceAccum >= _minTrailingSilence) return true;
        if (_sawSpeech && _utteranceAccum >= _maxUtterance) return true;
        return false;
    }

    public void Reset()
    {
        _silenceAccum = 0;
        _utteranceAccum = 0;
        _sawSpeech = false;
    }
}
