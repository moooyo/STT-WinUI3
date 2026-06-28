using Stt.Abstractions.Decoders;
using Stt.Core.Text;

namespace Stt.Core.Decoders;

/// <summary>Options for <see cref="NarDecoder"/> (spec §8.4).</summary>
public sealed record NarDecoderOptions
{
    /// <summary>CTC blank id.</summary>
    public int Blank { get; init; } = 0;

    /// <summary>
    /// Number of leading "query" output slots to drop (SenseVoice prepends 4: language / event /
    /// emotion / textnorm). Set 0 for plain Paraformer/CTC.
    /// </summary>
    public int QuerySlots { get; init; } = 4;

    /// <summary>Whether to strip <c>&lt;|...|&gt;</c> rich-transcription tags from the output.</summary>
    public bool StripTags { get; init; } = true;
}

/// <summary>
/// Offline non-autoregressive decoder for SenseVoice / Paraformer (spec §8.4, D4). Buffers
/// features until <see cref="InputFinished"/>, runs the model once, greedily CTC-collapses the
/// logits, drops the leading query slots, detokenizes via SentencePiece, and strips the
/// rich-transcription tags. Decode logic is exercised through an <see cref="IModelRunner"/> seam.
/// </summary>
public sealed class NarDecoder : IAsrDecoder
{
    private readonly IModelRunner _runner;
    private readonly TokenTable _tokens;
    private readonly NarDecoderOptions _opts;

    private readonly List<float> _featBuf = new();
    private int _featDim;
    private AsrResult _result = AsrResult.Empty;
    private bool _finished;

    public NarDecoder(IModelRunner runner, TokenTable tokens, NarDecoderOptions? options = null)
    {
        _runner = runner;
        _tokens = tokens;
        _opts = options ?? new NarDecoderOptions();
    }

    public DecoderCapabilities Capabilities =>
        DecoderCapabilities.Offline | DecoderCapabilities.Multilingual | DecoderCapabilities.Timestamps;

    public void Reset()
    {
        _featBuf.Clear();
        _featDim = 0;
        _result = AsrResult.Empty;
        _finished = false;
    }

    public bool AcceptFeatures(ReadOnlySpan<float> features, int numFrames, int featDim)
    {
        if (_finished) return false;
        _featDim = featDim;
        for (int i = 0; i < features.Length; i++) _featBuf.Add(features[i]);
        return true;
    }

    public void InputFinished()
    {
        if (_finished) return;
        _finished = true;

        if (_featBuf.Count == 0 || _featDim == 0)
        {
            _result = new AsrResult(string.Empty, Array.Empty<int>(), Array.Empty<float>(), IsFinal: true);
            return;
        }

        int numFrames = _featBuf.Count / _featDim;
        float[] logits = _runner.Run(_featBuf.ToArray(), numFrames, _featDim, out int outT, out int vocab);

        int[] ids = GreedyCtc.Decode(logits, outT, vocab, _opts.Blank);

        // Drop leading query slots (SenseVoice language/event/emotion/textnorm).
        ReadOnlySpan<int> kept = ids.Length > _opts.QuerySlots ? ids.AsSpan(_opts.QuerySlots) : ReadOnlySpan<int>.Empty;

        var pieces = new List<string>(kept.Length);
        foreach (int id in kept) pieces.Add(_tokens.Piece(id));

        string text = SentencePieceDetokenizer.Decode(pieces);
        if (_opts.StripTags)
            text = SpecialTagStripper.Strip(text).Clean;

        _result = new AsrResult(text, kept.ToArray(), Array.Empty<float>(), IsFinal: true);
    }

    public bool IsEndpoint() => false;   // offline decoder has no streaming endpoint

    public AsrResult GetResult() => _result;

    public void Dispose() { }
}
