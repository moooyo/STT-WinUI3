using Stt.Abstractions.Decoders;
using Stt.Core.Text;

namespace Stt.Core.Decoders;

/// <summary>
/// Streaming Zipformer transducer decoder (spec §8.2, D3). Buffers feature rows into fixed
/// <c>T</c>-frame chunks (advancing by <c>decode_chunk_len</c>), runs the encoder with cross-chunk
/// state, and feeds each encoder output frame to <see cref="TransducerGreedy"/>; the running
/// hypothesis is detokenized into a live partial. Endpointing is reported from trailing
/// non-emitting frames (the pipeline also gates on VAD silence). Final commitment happens in the
/// two-pass offline re-decode, so streaming results stay non-final until <see cref="InputFinished"/>.
/// </summary>
public sealed class TransducerDecoder : IAsrDecoder
{
    private readonly OrtStreamingSession _session;
    private readonly TransducerGreedy _greedy;
    private readonly TokenTable _tokens;
    private readonly int _chunkFrames;   // T
    private readonly int _chunkAdvance;  // decode_chunk_len
    private readonly double _frameShiftSec;
    private readonly double _endpointSilenceSec;

    private readonly List<float> _featBuf = new();
    private int _featDim;
    private int _framesSinceEmit;
    private bool _finished;
    private string _text = string.Empty;

    public TransducerDecoder(
        OrtStreamingSession session, TokenTable tokens,
        double frameShiftSeconds = 0.04, double endpointSilenceSeconds = 0.8)
    {
        _session = session;
        _tokens = tokens;
        var g = session.Geometry;
        _chunkFrames = g.T > 0 ? g.T : 0;
        _chunkAdvance = g.DecodeChunkLen > 0 ? g.DecodeChunkLen : Math.Max(1, _chunkFrames);
        _frameShiftSec = frameShiftSeconds;
        _endpointSilenceSec = endpointSilenceSeconds;
        _greedy = new TransducerGreedy(session.Predict, session.Join, blank: 0, contextSize: g.ContextSize);
    }

    public DecoderCapabilities Capabilities =>
        DecoderCapabilities.Streaming | DecoderCapabilities.PartialResults |
        DecoderCapabilities.Endpointing | DecoderCapabilities.Timestamps | DecoderCapabilities.Multilingual;

    public void Reset()
    {
        _featBuf.Clear();
        _featDim = 0;
        _framesSinceEmit = 0;
        _finished = false;
        _text = string.Empty;
        _session.ResetStates();
        _greedy.Reset();
    }

    public bool AcceptFeatures(ReadOnlySpan<float> features, int numFrames, int featDim)
    {
        if (_finished) return false;
        _featDim = featDim;
        for (int i = 0; i < features.Length; i++) _featBuf.Add(features[i]);

        bool advanced = false;
        // When a full chunk is buffered, encode + decode it.
        int chunk = _chunkFrames > 0 ? _chunkFrames : (_featBuf.Count / Math.Max(1, featDim));
        while (chunk > 0 && _featBuf.Count >= chunk * featDim)
        {
            advanced |= DecodeChunk(chunk, featDim);
            int advanceRows = Math.Min(_chunkAdvance, chunk);
            _featBuf.RemoveRange(0, advanceRows * featDim);
            if (_chunkFrames == 0) break; // dynamic: consume all at once
        }
        return advanced;
    }

    private bool DecodeChunk(int frames, int featDim)
    {
        var chunkArr = new float[frames * featDim];
        _featBuf.CopyTo(0, chunkArr, 0, chunkArr.Length);

        int before = _greedy.Tokens.Count;
        float[][] encFrames = _session.Encode(chunkArr, frames, featDim);
        foreach (var f in encFrames)
        {
            int countBefore = _greedy.Tokens.Count;
            _greedy.ProcessFrame(f);
            if (_greedy.Tokens.Count > countBefore) _framesSinceEmit = 0;
            else _framesSinceEmit++;
        }

        if (_greedy.Tokens.Count != before)
        {
            _text = Detokenize(_greedy.Tokens);
            return true;
        }
        return false;
    }

    public void InputFinished()
    {
        if (_finished) return;
        _finished = true;

        // Flush any remaining partial chunk (pad with zeros to the chunk length).
        if (_featDim > 0 && _featBuf.Count > 0)
        {
            int frames = _chunkFrames > 0 ? _chunkFrames : _featBuf.Count / _featDim;
            var chunkArr = new float[frames * _featDim];
            int copy = Math.Min(chunkArr.Length, _featBuf.Count);
            _featBuf.CopyTo(0, chunkArr, 0, copy);
            foreach (var f in _session.Encode(chunkArr, frames, _featDim))
                _greedy.ProcessFrame(f);
            _text = Detokenize(_greedy.Tokens);
        }
    }

    public bool IsEndpoint() => _framesSinceEmit * _frameShiftSec >= _endpointSilenceSec;

    public AsrResult GetResult() =>
        new(_text, _greedy.Tokens.ToArray(), Array.Empty<float>(), IsFinal: _finished);

    private string Detokenize(IReadOnlyList<int> ids)
    {
        var pieces = new List<string>(ids.Count);
        foreach (int id in ids) pieces.Add(_tokens.Piece(id));
        return SentencePieceDetokenizer.Decode(pieces);
    }

    public void Dispose() => _session.Dispose();
}
