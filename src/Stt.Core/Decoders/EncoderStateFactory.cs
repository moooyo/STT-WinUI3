using Microsoft.ML.OnnxRuntime;

namespace Stt.Core.Decoders;

/// <summary>
/// Builds the initial (zero) encoder state tensors for a streaming Zipformer transducer from its
/// metadata-derived geometry (spec §8.2). Dispatches on architecture: <b>v2</b> (zipformer2) has
/// per encoder layer 6 caches plus two global states (<c>m*6 + 2</c>); <b>v1</b> (legacy zipformer)
/// has per encoder <i>stack</i> 7 caches (len/avg/key/val/val2/conv1/conv2). Shapes follow
/// sherpa-onnx; nothing is hardcoded beyond the structural constants. Numerical correctness is
/// confirmed by aligning against sherpa-onnx before the streaming decoder is trusted.
/// </summary>
public static class EncoderStateFactory
{
    /// <summary>The ordered state specs: per-layer 6 caches grouped by layer, then embed + processed_lens.</summary>
    public static List<StateSpec> BuildSpecs(ZipformerGeometry g) =>
        g.Version == 1 ? BuildV1Specs(g) : BuildV2Specs(g);

    private static List<StateSpec> BuildV2Specs(ZipformerGeometry g)
    {
        var specs = new List<StateSpec>();
        int idx = 0;

        for (int e = 0; e < g.NumEncoderLayers.Length; e++)
        {
            int layers = g.NumEncoderLayers[e];
            int L = g.LeftContextLen[e];
            int keyDim = g.QueryHeadDims[e] * g.NumHeads[e];
            int valueDim = g.ValueHeadDims[e] * g.NumHeads[e];
            int nonlinDim = 3 * g.EncoderDims[e] / 4;
            // Streaming conv cache width is kernel/2 (= (kernel-1)/2 for the odd kernels icefall
            // uses), e.g. kernel 31 → 15, kernel 15 → 7 — verified against the sherpa-onnx export.
            int convCache = g.CnnModuleKernels[e] / 2;
            int encDim = g.EncoderDims[e];

            for (int j = 0; j < layers; j++)
            {
                specs.Add(new StateSpec($"cached_key_{idx}", new long[] { L, 1, keyDim }, StateDType.Float32));
                specs.Add(new StateSpec($"cached_nonlin_attn_{idx}", new long[] { 1, 1, L, nonlinDim }, StateDType.Float32));
                specs.Add(new StateSpec($"cached_val1_{idx}", new long[] { L, 1, valueDim }, StateDType.Float32));
                specs.Add(new StateSpec($"cached_val2_{idx}", new long[] { L, 1, valueDim }, StateDType.Float32));
                specs.Add(new StateSpec($"cached_conv1_{idx}", new long[] { 1, encDim, convCache }, StateDType.Float32));
                specs.Add(new StateSpec($"cached_conv2_{idx}", new long[] { 1, encDim, convCache }, StateDType.Float32));
                idx++;
            }
        }

        // Global embed cache from Conv2dSubsampling: (1, channels, 3, ((featDim-1)/2 - 1)/2).
        int embedF = (((g.FeatureDim - 1) / 2) - 1) / 2;
        specs.Add(new StateSpec("embed_states", new long[] { 1, g.EmbedChannels, 3, embedF }, StateDType.Float32));

        // processed_lens: int64 (1,).
        specs.Add(new StateSpec("processed_lens", new long[] { 1 }, StateDType.Int64));

        return specs;
    }

    /// <summary>
    /// Legacy Zipformer v1 (e.g. streaming bilingual zh-en 2023-02-20): per-stack caches, 7 tensors
    /// per encoder stack — len/avg/key/val/val2/conv1/conv2 — grouped by type then stack, matching
    /// sherpa-onnx <c>online-zipformer-transducer-model.cc</c>. No embed_states/processed_lens. Each
    /// tensor's leading dim is the stack's layer count; conv cache width is kernel-1; value dim is
    /// attention_dim/2. Verified against the model's ONNX input metadata.
    /// </summary>
    private static List<StateSpec> BuildV1Specs(ZipformerGeometry g)
    {
        int n = g.NumEncoderLayers.Length;
        long L(int i) => g.NumEncoderLayers[i];
        long lc(int i) => g.LeftContextLen[i];
        long enc(int i) => g.EncoderDims[i];
        long att(int i) => g.AttentionDims.Length > i ? g.AttentionDims[i] : g.EncoderDims[i] / 2;
        long conv(int i) => g.CnnModuleKernels[i] - 1;

        var specs = new List<StateSpec>();
        for (int i = 0; i < n; i++) specs.Add(new StateSpec($"cached_len_{i}", new long[] { L(i), 1 }, StateDType.Int64));
        for (int i = 0; i < n; i++) specs.Add(new StateSpec($"cached_avg_{i}", new long[] { L(i), 1, enc(i) }, StateDType.Float32));
        for (int i = 0; i < n; i++) specs.Add(new StateSpec($"cached_key_{i}", new long[] { L(i), lc(i), 1, att(i) }, StateDType.Float32));
        for (int i = 0; i < n; i++) specs.Add(new StateSpec($"cached_val_{i}", new long[] { L(i), lc(i), 1, att(i) / 2 }, StateDType.Float32));
        for (int i = 0; i < n; i++) specs.Add(new StateSpec($"cached_val2_{i}", new long[] { L(i), lc(i), 1, att(i) / 2 }, StateDType.Float32));
        for (int i = 0; i < n; i++) specs.Add(new StateSpec($"cached_conv1_{i}", new long[] { L(i), 1, enc(i), conv(i) }, StateDType.Float32));
        for (int i = 0; i < n; i++) specs.Add(new StateSpec($"cached_conv2_{i}", new long[] { L(i), 1, enc(i), conv(i) }, StateDType.Float32));
        return specs;
    }

    /// <summary>Allocate zero-initialized <see cref="OrtValue"/>s for each spec (ORT layer over <see cref="BuildSpecs"/>).</summary>
    public static List<OrtValue> BuildInitialStates(ZipformerGeometry g)
    {
        var values = new List<OrtValue>();
        foreach (StateSpec spec in BuildSpecs(g))
        {
            long count = 1;
            foreach (long d in spec.Shape) count *= d;
            OrtValue v = spec.DType == StateDType.Float32
                ? OrtValue.CreateTensorValueFromMemory(new float[count], spec.Shape)
                : OrtValue.CreateTensorValueFromMemory(new long[count], spec.Shape);
            values.Add(v);
        }
        return values;
    }
}
