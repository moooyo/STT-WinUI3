namespace Stt.Core.Audio;

/// <summary>
/// A fixed-capacity single-producer/single-consumer float ring buffer (spec §6: audio plumbing).
/// Not thread-safe for concurrent writers or concurrent readers; intended for one producer and
/// one consumer coordinated externally. Overflow truncates (returns the number actually written)
/// rather than blocking, matching the "real-time first" policy.
/// </summary>
public sealed class RingBuffer
{
    private readonly float[] _buf;
    private int _head;   // next read
    private int _tail;   // next write
    private int _count;

    public RingBuffer(int capacity)
    {
        if (capacity <= 0) throw new ArgumentOutOfRangeException(nameof(capacity));
        _buf = new float[capacity];
    }

    public int Capacity => _buf.Length;
    public int Count => _count;
    public int Free => _buf.Length - _count;

    /// <summary>Write up to <paramref name="src"/>.Length samples; returns the count actually written.</summary>
    public int Write(ReadOnlySpan<float> src)
    {
        int toWrite = Math.Min(src.Length, Free);
        for (int i = 0; i < toWrite; i++)
        {
            _buf[_tail] = src[i];
            _tail = _tail + 1 == _buf.Length ? 0 : _tail + 1;
        }
        _count += toWrite;
        return toWrite;
    }

    /// <summary>Read up to <paramref name="dst"/>.Length samples; returns the count actually read.</summary>
    public int Read(Span<float> dst)
    {
        int toRead = Math.Min(dst.Length, _count);
        for (int i = 0; i < toRead; i++)
        {
            dst[i] = _buf[_head];
            _head = _head + 1 == _buf.Length ? 0 : _head + 1;
        }
        _count -= toRead;
        return toRead;
    }

    public void Clear()
    {
        _head = _tail = _count = 0;
    }
}
