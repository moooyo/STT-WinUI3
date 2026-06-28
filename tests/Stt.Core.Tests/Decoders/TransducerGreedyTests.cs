using Stt.Core.Decoders;

namespace Stt.Core.Tests.Decoders;

public class TransducerGreedyTests
{
    // A joiner driven by a scripted argmax sequence; ignores its inputs.
    private sealed class ScriptedJoiner
    {
        private readonly Queue<int> _argmaxes;
        private readonly int _vocab;
        public int Calls { get; private set; }
        public ScriptedJoiner(int vocab, IEnumerable<int> argmaxSequence)
        {
            _vocab = vocab;
            _argmaxes = new Queue<int>(argmaxSequence);
        }
        public float[] Join(float[] enc, float[] dec)
        {
            Calls++;
            var logits = new float[_vocab];
            int best = _argmaxes.Count > 0 ? _argmaxes.Dequeue() : 0; // default to blank when exhausted
            logits[best] = 10f;
            return logits;
        }
    }

    [Fact]
    public void Emits_NonBlank_And_Allows_Consecutive_Tokens()
    {
        // frame0: blank
        // frame1: 5, blank
        // frame2: 5, 7, blank
        var joiner = new ScriptedJoiner(vocab: 8, argmaxSequence: new[] { 0, 5, 0, 5, 7, 0 });
        int predictorCalls = 0;
        Func<int[], float[]> predictor = _ => { predictorCalls++; return new float[4]; };

        var greedy = new TransducerGreedy(predictor, joiner.Join, blank: 0, contextSize: 2);
        int basePredictorCalls = greedy.PredictorCalls; // 1 from Reset

        greedy.ProcessFrame(new float[16]); // blank only
        greedy.ProcessFrame(new float[16]); // 5 then blank
        greedy.ProcessFrame(new float[16]); // 5, 7 then blank

        Assert.Equal(new[] { 5, 5, 7 }, greedy.Tokens);
        // predictor reruns once per emitted non-blank (3), plus the initial Reset call.
        Assert.Equal(basePredictorCalls + 3, greedy.PredictorCalls);
    }

    [Fact]
    public void All_Blank_Emits_Nothing()
    {
        var joiner = new ScriptedJoiner(vocab: 4, argmaxSequence: new[] { 0, 0, 0 });
        var greedy = new TransducerGreedy(_ => new float[2], joiner.Join, blank: 0, contextSize: 2);
        greedy.ProcessFrame(new float[8]);
        greedy.ProcessFrame(new float[8]);
        Assert.Empty(greedy.Tokens);
    }

    [Fact]
    public void Respects_Max_Symbols_Per_Frame()
    {
        // Never emits blank → would loop forever without the per-frame cap.
        var joiner = new ScriptedJoiner(vocab: 4, argmaxSequence: Enumerable.Repeat(3, 100));
        var greedy = new TransducerGreedy(_ => new float[2], joiner.Join, blank: 0, contextSize: 2, maxSymbolsPerFrame: 4);
        greedy.ProcessFrame(new float[8]);
        Assert.Equal(4, greedy.Tokens.Count); // capped
    }

    [Fact]
    public void Reset_Clears_Hypothesis()
    {
        var joiner = new ScriptedJoiner(vocab: 4, argmaxSequence: new[] { 2, 0 });
        var greedy = new TransducerGreedy(_ => new float[2], joiner.Join, blank: 0, contextSize: 2);
        greedy.ProcessFrame(new float[8]);
        Assert.NotEmpty(greedy.Tokens);
        greedy.Reset();
        Assert.Empty(greedy.Tokens);
    }
}
