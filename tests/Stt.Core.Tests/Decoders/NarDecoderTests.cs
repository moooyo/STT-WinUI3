using Stt.Core.Decoders;
using Stt.Core.Text;

namespace Stt.Core.Tests.Decoders;

public class GreedyCtcTests
{
    // Build row-major [T,V] logits where frame t has a peak at id `path[t]`.
    private static float[] Logits(int[] path, int v)
    {
        var l = new float[path.Length * v];
        for (int t = 0; t < path.Length; t++) l[t * v + path[t]] = 10f;
        return l;
    }

    [Fact]
    public void Collapses_Repeats_And_Drops_Blank()
    {
        // path a,a,blank,b,b  (a=1,b=2,blank=0) → [1,2]
        var l = Logits(new[] { 1, 1, 0, 2, 2 }, v: 3);
        Assert.Equal(new[] { 1, 2 }, GreedyCtc.Decode(l, T: 5, V: 3, blank: 0));
    }

    [Fact]
    public void All_Blank_Is_Empty()
    {
        var l = Logits(new[] { 0, 0, 0 }, v: 3);
        Assert.Empty(GreedyCtc.Decode(l, 3, 3, 0));
    }

    [Fact]
    public void Repeat_After_Blank_Is_Kept()
    {
        // a, blank, a → two separate 'a' (1,1) because the blank breaks the run.
        var l = Logits(new[] { 1, 0, 1 }, v: 2);
        Assert.Equal(new[] { 1, 1 }, GreedyCtc.Decode(l, 3, 2, 0));
    }
}

public class NarDecoderTests
{
    // A fake runner that returns a fixed [T,V] peak path regardless of features.
    private sealed class FakeRunner : IModelRunner
    {
        private readonly int[] _path;
        private readonly int _v;
        public FakeRunner(int[] path, int v) { _path = path; _v = v; }
        public float[] Run(float[] features, int numFrames, int featDim, out int outFrames, out int vocab)
        {
            outFrames = _path.Length;
            vocab = _v;
            var l = new float[_path.Length * _v];
            for (int t = 0; t < _path.Length; t++) l[t * _v + _path[t]] = 10f;
            return l;
        }
    }

    [Fact]
    public void Decodes_Collapse_Detokenize_Strip()
    {
        // vocab: 0=<blk>, 1=<|zh|>, 2=<|NEUTRAL|>, 3=<|Speech|>, 4=<|woitn|>, 5=你, 6=好, 7=▁world
        var tokens = TokenTable.Parse(new[]
        {
            "<blk> 0", "<|zh|> 1", "<|NEUTRAL|> 2", "<|Speech|> 3", "<|woitn|> 4", "你 5", "好 6", "▁world 7"
        });

        // path: 4 query slots (1,2,3,4) then 你,好,world with blanks/dups
        var path = new[] { 1, 2, 3, 4, 5, 5, 0, 6, 0, 7 };
        var runner = new FakeRunner(path, v: 8);

        var dec = new NarDecoder(runner, tokens, new NarDecoderOptions { QuerySlots = 4, Blank = 0, StripTags = true });
        dec.AcceptFeatures(new float[560], numFrames: 1, featDim: 560);
        dec.InputFinished();

        var result = dec.GetResult();
        Assert.True(result.IsFinal);
        Assert.Equal("你好 world", result.Text);
    }

    [Fact]
    public void Empty_Input_Yields_Empty_Final()
    {
        var tokens = TokenTable.Parse(new[] { "<blk> 0", "a 1" });
        var dec = new NarDecoder(new FakeRunner(new[] { 0 }, 2), tokens);
        dec.InputFinished();
        Assert.Equal(string.Empty, dec.GetResult().Text);
        Assert.True(dec.GetResult().IsFinal);
    }
}
