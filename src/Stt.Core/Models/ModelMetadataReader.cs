using Microsoft.ML.OnnxRuntime;

namespace Stt.Core.Models;

/// <summary>
/// Reads the signals needed for family detection + validation from ONNX metadata (spec §10.1).
/// The pure <see cref="Read"/> overload works on already-extracted dictionaries so it is unit
/// testable without ORT; <see cref="FromSession"/> pulls them from a live
/// <see cref="InferenceSession"/>.
/// </summary>
public static class ModelMetadataReader
{
    private static readonly string[] FeatureInputNames =
        { "x", "speech", "feats", "feature", "features", "audio_signal", "mel", "input_features", "wav" };

    /// <summary>
    /// Build a <see cref="ModelProbe"/> from a custom-metadata map and a map of input tensor name
    /// → dimensions (with -1 for dynamic axes).
    /// </summary>
    public static ModelProbe Read(
        IReadOnlyDictionary<string, string> meta,
        IReadOnlyDictionary<string, int[]> inputDims)
    {
        string modelType = FirstNonEmpty(meta, "model_type", "model", "framework", "architecture");

        var (name, dims) = PickFeatureInput(inputDims);
        var (featDim, layout) = InterpretDims(dims);

        int? nMels = TryGetInt(meta, "n_mels") ?? TryGetInt(meta, "feat_dim") ?? TryGetInt(meta, "feature_dim");
        int vocab = TryGetInt(meta, "vocab_size") ?? TryGetInt(meta, "vocab") ?? 0;

        return new ModelProbe
        {
            ModelType = modelType,
            FeatureDim = featDim,
            NMels = nMels,
            Layout = layout,
            VocabSize = vocab,
            Metadata = meta,
        };
    }

    /// <summary>Extract metadata + the feature input dims from a live session.</summary>
    public static ModelProbe FromSession(InferenceSession session)
    {
        var meta = session.ModelMetadata.CustomMetadataMap;
        var inputDims = new Dictionary<string, int[]>();
        foreach (var kv in session.InputMetadata)
            inputDims[kv.Key] = kv.Value.Dimensions;
        return Read(meta, inputDims);
    }

    private static (string Name, int[] Dims) PickFeatureInput(IReadOnlyDictionary<string, int[]> inputs)
    {
        // Prefer a known feature-input name with rank 3.
        foreach (string n in FeatureInputNames)
            if (inputs.TryGetValue(n, out var d) && d.Length == 3)
                return (n, d);

        // Otherwise the first rank-3 input.
        foreach (var kv in inputs)
            if (kv.Value.Length == 3)
                return (kv.Key, kv.Value);

        // Fallback: first rank-2 (raw audio [N, samples]) or anything.
        foreach (var kv in inputs)
            if (kv.Value.Length == 2)
                return (kv.Key, kv.Value);

        return inputs.Count > 0 ? (inputs.First().Key, inputs.First().Value) : ("", Array.Empty<int>());
    }

    private static (int FeatDim, FeatureLayout Layout) InterpretDims(int[] dims)
    {
        if (dims.Length != 3) return (0, FeatureLayout.Unknown);

        int a = dims[1], b = dims[2];

        // Whisper layout: time axis fixed at 3000 → [N, n_mels, 3000].
        if (b == 3000) return (a, FeatureLayout.MelMid);

        // fbank / LFR layout: [N, T, C] with time dynamic (-1) and feature dim fixed last.
        if (b > 0) return (b, FeatureLayout.TimeLast);

        // Last axis dynamic but middle fixed → treat middle as feature dim (rare re-export case).
        if (a > 0) return (a, FeatureLayout.TimeLast);

        return (0, FeatureLayout.Unknown);
    }

    private static string FirstNonEmpty(IReadOnlyDictionary<string, string> meta, params string[] keys)
    {
        foreach (string k in keys)
            if (meta.TryGetValue(k, out var v) && !string.IsNullOrWhiteSpace(v))
                return v;
        return "";
    }

    private static int? TryGetInt(IReadOnlyDictionary<string, string> meta, string key) =>
        meta.TryGetValue(key, out var v) && int.TryParse(v, out int i) ? i : null;
}
