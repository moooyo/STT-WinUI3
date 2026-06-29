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

        // encoder 0: L=64, keyDim=32*4=128, valueDim=12*4=48, nonlin=3*192/4=144, conv=31/2=15, encDim=192
        Assert.Equal(new long[] { 64, 1, 128 }, specs[0].Shape);   // cached_key
        Assert.Equal(new long[] { 1, 1, 64, 144 }, specs[1].Shape); // cached_nonlin_attn
        Assert.Equal(new long[] { 64, 1, 48 }, specs[2].Shape);     // cached_val1
        Assert.Equal(new long[] { 64, 1, 48 }, specs[3].Shape);     // cached_val2
        Assert.Equal(new long[] { 1, 192, 15 }, specs[4].Shape);    // cached_conv1 (sherpa: kernel/2)
        Assert.Equal(new long[] { 1, 192, 15 }, specs[5].Shape);    // cached_conv2 (sherpa: kernel/2)
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

    // Legacy v1 geometry — the streaming bilingual zh-en 2023-02-20 model. Per-stack 7 caches.
    private static readonly Dictionary<string, string> MetaV1 = new()
    {
        ["model_type"] = "zipformer",
        ["num_encoder_layers"] = "2,4,3,2,4",
        ["encoder_dims"] = "384,384,384,384,384",
        ["attention_dims"] = "192,192,192,192,192",
        ["cnn_module_kernels"] = "31,31,31,31,31",
        ["left_context_len"] = "64,32,16,8,32",
        ["T"] = "39",
        ["decode_chunk_len"] = "32",
    };

    [Fact]
    public void V1_Is_Detected_And_Has_7_Caches_Per_Stack()
    {
        var geo = ZipformerGeometry.FromMetadata(MetaV1);
        Assert.Equal(1, geo.Version);
        var specs = EncoderStateFactory.BuildSpecs(geo);
        Assert.Equal(5 * 7, specs.Count);                  // 5 stacks × 7 caches, no global states
        Assert.DoesNotContain(specs, s => s.Name == "embed_states");
    }

    [Fact]
    public void V1_Cache_Shapes_Match_Sherpa_Onnx_Inputs()
    {
        var specs = EncoderStateFactory.BuildSpecs(ZipformerGeometry.FromMetadata(MetaV1));
        StateSpec S(string n) => specs.First(s => s.Name == n);
        // stack 0: L=2, lc=64, enc=384, att=192, kernel-1=30 — verified against encoder.onnx metadata.
        Assert.Equal(new long[] { 2, 1 }, S("cached_len_0").Shape);
        Assert.Equal(StateDType.Int64, S("cached_len_0").DType);
        Assert.Equal(new long[] { 2, 1, 384 }, S("cached_avg_0").Shape);
        Assert.Equal(new long[] { 2, 64, 1, 192 }, S("cached_key_0").Shape);
        Assert.Equal(new long[] { 2, 64, 1, 96 }, S("cached_val_0").Shape);
        Assert.Equal(new long[] { 2, 64, 1, 96 }, S("cached_val2_0").Shape);
        Assert.Equal(new long[] { 2, 1, 384, 30 }, S("cached_conv1_0").Shape);
        Assert.Equal(new long[] { 4, 32, 1, 192 }, S("cached_key_1").Shape);  // stack 1: L=4, lc=32
    }
}
