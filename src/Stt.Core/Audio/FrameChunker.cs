namespace Stt.Core.Audio;

/// <summary>
/// Accumulates a stream of samples and emits fixed-size windows with a configurable hop,
/// supporting overlap (window &gt; hop). Used to feed the VAD fixed 512-sample windows and to
/// frame features. Keeps an internal buffer of the not-yet-consumed tail.
/// </summary>
public sealed class FrameChunker
{
    private readonly int _window;
    private readonly int _hop;
    private float[] _buf;
    private int _count;

    public FrameChunker(int windowSize, int hopSize)
    {
        if (windowSize <= 0) throw new ArgumentOutOfRangeException(nameof(windowSize));
        if (hopSize <= 0 || hopSize > windowSize) throw new ArgumentOutOfRangeException(nameof(hopSize));
        _window = windowSize;
        _hop = hopSize;
        _buf = new float[Math.Max(windowSize * 2, 1024)];
    }

    public int WindowSize => _window;
    public int HopSize => _hop;

    /// <summary>Number of buffered samples not yet emitted as a full window.</summary>
    public int Buffered => _count;

    public void Push(ReadOnlySpan<float> samples)
    {
        EnsureCapacity(_count + samples.Length);
        samples.CopyTo(_buf.AsSpan(_count));
        _count += samples.Length;
    }

    /// <summary>
    /// Emit the next full window if available. The window is copied out; the internal buffer
    /// advances by <see cref="HopSize"/> (so overlap is retained when window &gt; hop).
    /// </summary>
    public bool TryGetWindow(out float[] window)
    {
        if (_count < _window)
        {
            window = Array.Empty<float>();
            return false;
        }

        window = new float[_window];
        Array.Copy(_buf, 0, window, 0, _window);

        // Slide left by hop.
        int remaining = _count - _hop;
        Array.Copy(_buf, _hop, _buf, 0, remaining);
        _count = remaining;
        return true;
    }

    public void Reset() => _count = 0;

    private void EnsureCapacity(int needed)
    {
        if (needed <= _buf.Length) return;
        int newCap = _buf.Length * 2;
        while (newCap < needed) newCap *= 2;
        Array.Resize(ref _buf, newCap);
    }
}
