namespace Stt.Core.Decoders;

/// <summary>
/// Greedy RNN-T / Zipformer transducer decode logic (spec §8.2), decoupled from ORT so the state
/// management is unit testable. Per encoder frame it evaluates the joiner; a non-blank argmax is
/// appended to the hypothesis and the predictor is rerun to refresh <c>decoder_out</c> (so
/// consecutive identical tokens are allowed, unlike CTC); a blank advances to the next frame.
/// The predictor takes the last <c>context_size</c> tokens; <c>decoder_out</c> is cached across
/// chunks. The encoder/decoder/joiner forward passes are injected as delegates.
/// </summary>
public sealed class TransducerGreedy
{
    private readonly Func<int[], float[]> _predictor;      // context tokens → decoder_out
    private readonly Func<float[], float[], float[]> _joiner; // (encoderFrame, decoderOut) → logits[vocab]
    private readonly int _blank;
    private readonly int _contextSize;
    private readonly int _maxSymPerFrame;

    private readonly List<int> _hyp = new();
    private float[] _decoderOut = Array.Empty<float>();

    public TransducerGreedy(
        Func<int[], float[]> predictor,
        Func<float[], float[], float[]> joiner,
        int blank = 0,
        int contextSize = 2,
        int maxSymbolsPerFrame = 5)
    {
        _predictor = predictor;
        _joiner = joiner;
        _blank = blank;
        _contextSize = contextSize;
        _maxSymPerFrame = maxSymbolsPerFrame;
        Reset();
    }

    /// <summary>Decoded token ids (excluding the initial blank context).</summary>
    public IReadOnlyList<int> Tokens => _hyp.Skip(_contextSize).ToList();

    public int PredictorCalls { get; private set; }

    public void Reset()
    {
        _hyp.Clear();
        for (int i = 0; i < _contextSize; i++) _hyp.Add(_blank);
        PredictorCalls = 0;
        _decoderOut = _predictor(CurrentContext());
        PredictorCalls++;
    }

    /// <summary>Process one encoder output frame, emitting zero or more tokens.</summary>
    public void ProcessFrame(float[] encoderFrame)
    {
        int emitted = 0;
        while (emitted < _maxSymPerFrame)
        {
            float[] logits = _joiner(encoderFrame, _decoderOut);
            int best = ArgMax(logits);
            if (best == _blank) break;

            _hyp.Add(best);
            _decoderOut = _predictor(CurrentContext());
            PredictorCalls++;
            emitted++;
        }
    }

    private int[] CurrentContext()
    {
        var ctx = new int[_contextSize];
        int start = _hyp.Count - _contextSize;
        for (int i = 0; i < _contextSize; i++) ctx[i] = _hyp[start + i];
        return ctx;
    }

    private static int ArgMax(float[] row)
    {
        int best = 0;
        float bestVal = row.Length > 0 ? row[0] : 0f;
        for (int i = 1; i < row.Length; i++)
            if (row[i] > bestVal) { bestVal = row[i]; best = i; }
        return best;
    }
}
