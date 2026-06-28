namespace Stt.Abstractions.Decoders;

/// <summary>
/// What a decoder can do. Drives legal pipeline combinations and UI validation (spec §5.2,
/// §11): the first pass requires <see cref="Streaming"/>; two-pass/offline requires a decoder
/// with <see cref="Offline"/> for the second pass.
/// </summary>
[Flags]
public enum DecoderCapabilities
{
    None = 0,
    Streaming = 1,
    Offline = 2,
    PartialResults = 4,
    Endpointing = 8,
    Timestamps = 16,
    Multilingual = 32
}
