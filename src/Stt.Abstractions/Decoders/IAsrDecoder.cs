namespace Stt.Abstractions.Decoders;

/// <summary>
/// A streaming or offline ASR decoder (spec §5.1, §8.1). Streaming decoders consume features
/// incrementally and expose <see cref="IsEndpoint"/>; offline decoders buffer until
/// <see cref="InputFinished"/> then decode. The per-family decode loop and internal state are
/// implementation details and never leak through this interface.
/// </summary>
public interface IAsrDecoder : IDisposable
{
    /// <summary>What this decoder supports (drives legal pipeline combinations).</summary>
    DecoderCapabilities Capabilities { get; }

    /// <summary>Clear hypothesis + internal state for a new utterance.</summary>
    void Reset();

    /// <summary>
    /// Accept a chunk of features (row-major <c>[numFrames, featDim]</c>). Returns true if the
    /// hypothesis advanced. Offline decoders buffer; streaming decoders decode incrementally.
    /// </summary>
    bool AcceptFeatures(ReadOnlySpan<float> features, int numFrames, int featDim);

    /// <summary>Signal end of input (offline decoders run here).</summary>
    void InputFinished();

    /// <summary>Streaming only: true when the endpoint rule has fired. Always false for offline.</summary>
    bool IsEndpoint();

    /// <summary>Current best hypothesis (partial for streaming, final after <see cref="InputFinished"/>).</summary>
    AsrResult GetResult();
}
