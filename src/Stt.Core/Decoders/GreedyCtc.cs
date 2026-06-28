namespace Stt.Core.Decoders;

/// <summary>
/// Greedy CTC decoding (spec §8.3, §8.4): per-frame argmax over the vocabulary, then collapse
/// consecutive duplicate ids and drop the blank id. Shared by the streaming CTC path and the
/// offline NAR (SenseVoice/Paraformer) path.
/// </summary>
public static class GreedyCtc
{
    /// <summary>
    /// Decode row-major <c>[T, V]</c> log-probs (or logits — argmax is invariant to monotonic
    /// transforms) into a collapsed token id sequence.
    /// </summary>
    public static int[] Decode(ReadOnlySpan<float> logProbs, int T, int V, int blank = 0)
    {
        var outIds = new List<int>(T);
        int prev = -1;
        for (int t = 0; t < T; t++)
        {
            int best = ArgMax(logProbs.Slice(t * V, V));
            if (best != blank && best != prev)
                outIds.Add(best);
            prev = best;
        }
        return outIds.ToArray();
    }

    /// <summary>Per-frame argmax ids without collapsing (for alignment/timestamps).</summary>
    public static int[] ArgMaxPath(ReadOnlySpan<float> logProbs, int T, int V)
    {
        var path = new int[T];
        for (int t = 0; t < T; t++) path[t] = ArgMax(logProbs.Slice(t * V, V));
        return path;
    }

    private static int ArgMax(ReadOnlySpan<float> row)
    {
        int best = 0;
        float bestVal = row.Length > 0 ? row[0] : 0f;
        for (int i = 1; i < row.Length; i++)
        {
            if (row[i] > bestVal) { bestVal = row[i]; best = i; }
        }
        return best;
    }
}
