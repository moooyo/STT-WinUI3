namespace Stt.Core.Decoders;

/// <summary>
/// Streaming Zipformer transducer geometry parsed from ONNX <c>metadata_props</c> (spec §8.2). The
/// comma-separated arrays describe each encoder stack; per-cache shapes are derived from them (never
/// hardcoded). Two architectures: <b>v2</b> (zipformer2, per-layer attention via query/value head
/// dims + num_heads) and <b>v1</b> (legacy zipformer, per-stack attention_dims), per sherpa-onnx
/// <c>online-zipformer2-transducer-model.cc</c> / <c>online-zipformer-transducer-model.cc</c> +
/// icefall <c>export-onnx-streaming.py</c>.
/// </summary>
public sealed record ZipformerGeometry
{
    public required int[] NumEncoderLayers { get; init; }
    public required int[] EncoderDims { get; init; }
    public required int[] CnnModuleKernels { get; init; }
    public required int[] LeftContextLen { get; init; }

    /// <summary>v2 attention geometry; empty for v1 (which carries <see cref="AttentionDims"/> instead).</summary>
    public int[] QueryHeadDims { get; init; } = Array.Empty<int>();
    public int[] ValueHeadDims { get; init; } = Array.Empty<int>();
    public int[] NumHeads { get; init; } = Array.Empty<int>();

    /// <summary>v1 attention dim per stack; empty for v2.</summary>
    public int[] AttentionDims { get; init; } = Array.Empty<int>();

    /// <summary>1 = legacy zipformer (per-stack 7 caches), 2 = zipformer2 (per-layer 6 + 2 global).</summary>
    public int Version { get; init; } = 2;

    public int T { get; init; }
    public int DecodeChunkLen { get; init; }
    public int ContextSize { get; init; } = 2;
    public int VocabSize { get; init; }

    /// <summary>Feature dimension feeding the Conv2dSubsampling embed (for the embed_states shape).</summary>
    public int FeatureDim { get; init; } = 80;

    /// <summary>Conv2dSubsampling output channels (Zipformer2 default 128; overridable from metadata).</summary>
    public int EmbedChannels { get; init; } = 128;

    /// <summary>Total encoder layers across all stacks (m). The v2 state count is <c>m*6 + 2</c>.</summary>
    public int TotalLayers => Sum(NumEncoderLayers);

    public static ZipformerGeometry FromMetadata(IReadOnlyDictionary<string, string> meta)
    {
        int[] Arr(string key) => ParseIntArray(Require(meta, key));
        int[] ArrOpt(string key) => meta.TryGetValue(key, out var v) && !string.IsNullOrWhiteSpace(v)
            ? ParseIntArray(v) : Array.Empty<int>();
        int Scalar(string key, int dflt = 0) =>
            meta.TryGetValue(key, out var v) && int.TryParse(v, out int i) ? i : dflt;

        // zipformer2 → v2 (per-layer caches); legacy zipformer → v1 (per-stack caches). When the
        // model_type is unstated, attention_dims (v1) vs num_heads (v2) disambiguate. The 80-dim
        // fbank front-end is shared; only the encoder state geometry differs.
        string mt = meta.TryGetValue("model_type", out var t) ? t.ToLowerInvariant() : "";
        bool hasV1 = meta.ContainsKey("attention_dims");
        bool hasV2 = meta.ContainsKey("num_heads") || meta.ContainsKey("query_head_dims");
        int version = mt.Contains("zipformer2") ? 2 : mt == "zipformer" ? 1 : (hasV1 && !hasV2) ? 1 : 2;

        return new ZipformerGeometry
        {
            Version = version,
            NumEncoderLayers = Arr("num_encoder_layers"),
            EncoderDims = Arr("encoder_dims"),
            CnnModuleKernels = Arr("cnn_module_kernels"),
            LeftContextLen = Arr("left_context_len"),
            QueryHeadDims = version == 2 ? Arr("query_head_dims") : ArrOpt("query_head_dims"),
            ValueHeadDims = version == 2 ? Arr("value_head_dims") : ArrOpt("value_head_dims"),
            NumHeads = version == 2 ? Arr("num_heads") : ArrOpt("num_heads"),
            AttentionDims = version == 1 ? Arr("attention_dims") : ArrOpt("attention_dims"),
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
