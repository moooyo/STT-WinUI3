using Stt.Core.Decoders;

namespace Stt.Core.Tests.Decoders;

public class EncoderStateFactoryTests
{
    // Representative icefall streaming Zipformer2 geometry.
    private static readonly Dictionary<string, string> Meta = new()
    {
        ["num_encoder_layers"] = "2,2,3,4,3,2",     // m = 16
        ["encoder_dims"] = "192,256,384,512,384,256",
        ["cnn_module_kernels"] = "31,31,15,15,15,31",
        ["left_context_len"] = "64,32,16,8,16,32",
        ["query_head_dims"] = "32,32,32,32,32,32",
        ["value_head_dims"] = "12,12,12,12,12,12",
        ["num_heads"] = "4,4,4,8,4,4",
        ["T"] = "39",
        ["decode_chunk_len"] = "32",
        ["context_size"] = "2",
        ["vocab_size"] = "6254",
        ["feature_dim"] = "80",
    };

    [Fact]
    public void State_Count_Is_m_times_6_plus_2()
    {
        var geo = ZipformerGeometry.FromMetadata(Meta);
        Assert.Equal(16, geo.TotalLayers);

        var specs = EncoderStateFactory.BuildSpecs(geo);
        Assert.Equal(16 * 6 + 2, specs.Count);
    }

    [Fact]
    public void First_Layer_Cache_Shapes_Match_Sherpa_Formulas()
    {
        var geo = ZipformerGeometry.FromMetadata(Meta);
        var specs = EncoderStateFactory.BuildSpecs(geo);

        // encoder 0: L=64, keyDim=32*4=128, valueDim=12*4=48, nonlin=3*192/4=144, conv=31-1=30, encDim=192
        Assert.Equal(new long[] { 64, 1, 128 }, specs[0].Shape);   // cached_key
        Assert.Equal(new long[] { 1, 1, 64, 144 }, specs[1].Shape); // cached_nonlin_attn
        Assert.Equal(new long[] { 64, 1, 48 }, specs[2].Shape);     // cached_val1
        Assert.Equal(new long[] { 64, 1, 48 }, specs[3].Shape);     // cached_val2
        Assert.Equal(new long[] { 1, 192, 30 }, specs[4].Shape);    // cached_conv1
        Assert.Equal(new long[] { 1, 192, 30 }, specs[5].Shape);    // cached_conv2
    }

    [Fact]
    public void Global_States_Are_Embed_And_ProcessedLens()
    {
        var geo = ZipformerGeometry.FromMetadata(Meta);
        var specs = EncoderStateFactory.BuildSpecs(geo);

        var embed = specs[^2];
        var processed = specs[^1];

        Assert.Equal("embed_states", embed.Name);
        // (1, 128, 3, ((80-1)/2 - 1)/2) = (1,128,3,19)
        Assert.Equal(new long[] { 1, 128, 3, 19 }, embed.Shape);
        Assert.Equal(StateDType.Float32, embed.DType);

        Assert.Equal("processed_lens", processed.Name);
        Assert.Equal(new long[] { 1 }, processed.Shape);
        Assert.Equal(StateDType.Int64, processed.DType);
    }

    [Fact]
    public void Missing_Required_Key_Fails_Loud()
    {
        var bad = new Dictionary<string, string>(Meta);
        bad.Remove("num_heads");
        Assert.Throws<InvalidOperationException>(() => ZipformerGeometry.FromMetadata(bad));
    }

    [Fact]
    public void BuildInitialStates_Allocates_All_Tensors()
    {
        var geo = ZipformerGeometry.FromMetadata(Meta);
        var states = EncoderStateFactory.BuildInitialStates(geo);
        try
        {
            Assert.Equal(16 * 6 + 2, states.Count);
            Assert.All(states, s => Assert.NotNull(s));
        }
        finally
        {
            foreach (var s in states) s.Dispose();
        }
    }
}
