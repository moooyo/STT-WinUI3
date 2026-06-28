namespace Stt.Core.Decoders;

/// <summary>
/// Streaming Zipformer2 transducer geometry parsed from ONNX <c>metadata_props</c> (spec §8.2).
/// The comma-separated arrays describe each of the <c>num_encoders</c> encoder stacks; per-layer
/// cache shapes are derived from them (never hardcoded), per sherpa-onnx
/// <c>online-zipformer2-transducer-model.cc</c> + icefall <c>export-onnx-streaming.py</c>.
/// </summary>
public sealed record ZipformerGeometry
{
    public required int[] NumEncoderLayers { get; init; }
    public required int[] EncoderDims { get; init; }
    public required int[] CnnModuleKernels { get; init; }
    public required int[] LeftContextLen { get; init; }
    public required int[] QueryHeadDims { get; init; }
    public required int[] ValueHeadDims { get; init; }
    public required int[] NumHeads { get; init; }

    public int T { get; init; }
    public int DecodeChunkLen { get; init; }
    public int ContextSize { get; init; } = 2;
    public int VocabSize { get; init; }

    /// <summary>Feature dimension feeding the Conv2dSubsampling embed (for the embed_states shape).</summary>
    public int FeatureDim { get; init; } = 80;

    /// <summary>Conv2dSubsampling output channels (Zipformer2 default 128; overridable from metadata).</summary>
    public int EmbedChannels { get; init; } = 128;

    /// <summary>Total encoder layers across all stacks (m). The state count is <c>m*6 + 2</c>.</summary>
    public int TotalLayers => Sum(NumEncoderLayers);

    public static ZipformerGeometry FromMetadata(IReadOnlyDictionary<string, string> meta)
    {
        int[] Arr(string key) => ParseIntArray(Require(meta, key));
        int Scalar(string key, int dflt = 0) =>
            meta.TryGetValue(key, out var v) && int.TryParse(v, out int i) ? i : dflt;

        return new ZipformerGeometry
        {
            NumEncoderLayers = Arr("num_encoder_layers"),
            EncoderDims = Arr("encoder_dims"),
            CnnModuleKernels = Arr("cnn_module_kernels"),
            LeftContextLen = Arr("left_context_len"),
            QueryHeadDims = Arr("query_head_dims"),
            ValueHeadDims = Arr("value_head_dims"),
            NumHeads = Arr("num_heads"),
            T = Scalar("T"),
            DecodeChunkLen = Scalar("decode_chunk_len"),
            ContextSize = Scalar("context_size", 2),
            VocabSize = Scalar("vocab_size"),
            FeatureDim = Scalar("feature_dim", 80),
            EmbedChannels = Scalar("embed_dim", 128),
        };
    }

    private static string Require(IReadOnlyDictionary<string, string> meta, string key) =>
        meta.TryGetValue(key, out var v) && !string.IsNullOrWhiteSpace(v)
            ? v
            : throw new InvalidOperationException($"Streaming Zipformer metadata missing required key '{key}' (spec §8.2).");

    private static int[] ParseIntArray(string s)
    {
        string[] parts = s.Split(new[] { ',', ' ' }, StringSplitOptions.RemoveEmptyEntries);
        var a = new int[parts.Length];
        for (int i = 0; i < parts.Length; i++) a[i] = int.Parse(parts[i]);
        return a;
    }

    private static int Sum(int[] a) { int s = 0; foreach (int x in a) s += x; return s; }
}

/// <summary>A single encoder state tensor: its shape and element type (spec §8.2).</summary>
public sealed record StateSpec(string Name, long[] Shape, StateDType DType);

public enum StateDType { Float32, Int64 }
