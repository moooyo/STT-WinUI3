using System.Buffers.Binary;

namespace Stt.Core.Audio;

/// <summary>
/// Minimal RIFF/WAVE reader and writer for headless test fixtures and file-based capture.
/// Reads 16-bit PCM (format tag 1) and 32-bit IEEE float (format tag 3); writes 16-bit PCM.
/// Returns interleaved samples in [-1, 1] float along with sample rate and channel count.
/// </summary>
public static class WavIo
{
    public readonly record struct WavData(float[] Interleaved, int SampleRate, int Channels);

    public static WavData ReadPcm(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        return ParsePcm(bytes);
    }

    public static WavData ParsePcm(ReadOnlySpan<byte> bytes)
    {
        if (bytes.Length < 12 || !bytes[..4].SequenceEqual("RIFF"u8) || !bytes.Slice(8, 4).SequenceEqual("WAVE"u8))
            throw new InvalidDataException("Not a RIFF/WAVE file.");

        int pos = 12;
        int fmtTag = 0, channels = 0, sampleRate = 0, bitsPerSample = 0;
        ReadOnlySpan<byte> data = default;
        bool haveFmt = false, haveData = false;

        while (pos + 8 <= bytes.Length)
        {
            ReadOnlySpan<byte> id = bytes.Slice(pos, 4);
            int size = BinaryPrimitives.ReadInt32LittleEndian(bytes.Slice(pos + 4, 4));
            int body = pos + 8;
            if (body + size > bytes.Length) size = bytes.Length - body; // tolerate truncated size

            if (id.SequenceEqual("fmt "u8))
            {
                fmtTag = BinaryPrimitives.ReadUInt16LittleEndian(bytes.Slice(body, 2));
                channels = BinaryPrimitives.ReadUInt16LittleEndian(bytes.Slice(body + 2, 2));
                sampleRate = BinaryPrimitives.ReadInt32LittleEndian(bytes.Slice(body + 4, 4));
                bitsPerSample = BinaryPrimitives.ReadUInt16LittleEndian(bytes.Slice(body + 14, 2));
                haveFmt = true;
            }
            else if (id.SequenceEqual("data"u8))
            {
                data = bytes.Slice(body, size);
                haveData = true;
            }

            pos = body + size + (size & 1); // chunks are word-aligned
        }

        if (!haveFmt || !haveData) throw new InvalidDataException("Missing fmt or data chunk.");

        float[] samples = (fmtTag, bitsPerSample) switch
        {
            (1, 16) => DecodePcm16(data),
            (3, 32) => DecodeFloat32(data),
            (0xFFFE, 16) => DecodePcm16(data), // WAVE_FORMAT_EXTENSIBLE, 16-bit PCM
            _ => throw new NotSupportedException($"Unsupported WAV format tag={fmtTag}, bits={bitsPerSample}.")
        };

        return new WavData(samples, sampleRate, channels);
    }

    private static float[] DecodePcm16(ReadOnlySpan<byte> data)
    {
        int n = data.Length / 2;
        var outBuf = new float[n];
        for (int i = 0; i < n; i++)
        {
            short s = BinaryPrimitives.ReadInt16LittleEndian(data.Slice(i * 2, 2));
            outBuf[i] = s / 32768f;
        }
        return outBuf;
    }

    private static float[] DecodeFloat32(ReadOnlySpan<byte> data)
    {
        int n = data.Length / 4;
        var outBuf = new float[n];
        for (int i = 0; i < n; i++)
            outBuf[i] = BinaryPrimitives.ReadSingleLittleEndian(data.Slice(i * 4, 4));
        return outBuf;
    }

    /// <summary>Write mono float samples as a 16-bit PCM WAV at the given sample rate.</summary>
    public static void WritePcm16(string path, ReadOnlySpan<float> mono, int sampleRate)
    {
        const int channels = 1;
        int dataBytes = mono.Length * 2;
        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write);
        using var bw = new BinaryWriter(fs);

        bw.Write("RIFF"u8); bw.Write(36 + dataBytes); bw.Write("WAVE"u8);
        bw.Write("fmt "u8); bw.Write(16);            // PCM fmt chunk size
        bw.Write((ushort)1);                          // PCM
        bw.Write((ushort)channels);
        bw.Write(sampleRate);
        bw.Write(sampleRate * channels * 2);          // byte rate
        bw.Write((ushort)(channels * 2));             // block align
        bw.Write((ushort)16);                         // bits per sample
        bw.Write("data"u8); bw.Write(dataBytes);
        foreach (float f in mono)
        {
            // Scale by 32768 to match the ÷32768 read (true round-trip), clamp to int16 range.
            int s = (int)MathF.Round(Math.Clamp(f, -1f, 1f) * 32768f);
            s = Math.Clamp(s, short.MinValue, short.MaxValue);
            bw.Write((short)s);
        }
    }
}
