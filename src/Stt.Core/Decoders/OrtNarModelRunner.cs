using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Stt.Core.Decoders;

/// <summary>Options for <see cref="OrtNarModelRunner"/> — SenseVoice query ids (spec §8.4, §10.1).</summary>
public sealed record OrtNarRunnerOptions
{
    /// <summary>Language query id (SenseVoice: auto=0, zh, en, ... — read from metadata when available).</summary>
    public long LanguageId { get; init; } = 0;

    /// <summary>Text-norm/ITN query id (SenseVoice withitn/woitn).</summary>
    public long TextNormId { get; init; } = 15;
}

/// <summary>
/// ORT-backed <see cref="IModelRunner"/> for SenseVoice / Paraformer NAR models (spec §8.4). It
/// adapts to the model's declared input names: the 3-D float feature input receives the features,
/// an input whose name contains "len" receives the frame count, and "language"/"textnorm" inputs
/// receive the configured query ids. The first 3-D float output is taken as the logits. This
/// tolerance covers the common SenseVoice/Paraformer export variants without hardcoding one layout.
/// </summary>
public sealed class OrtNarModelRunner : IModelRunner, IDisposable
{
    private readonly InferenceSession _session;
    private readonly OrtNarRunnerOptions _opts;

    public OrtNarModelRunner(InferenceSession session, OrtNarRunnerOptions? options = null)
    {
        _session = session;
        _opts = options ?? new OrtNarRunnerOptions();
    }

    public float[] Run(float[] features, int numFrames, int featDim, out int outFrames, out int vocab)
    {
        var inputs = new List<NamedOnnxValue>();

        foreach (var kv in _session.InputMetadata)
        {
            string name = kv.Key;
            string lower = name.ToLowerInvariant();
            int rank = kv.Value.Dimensions.Length;

            if (rank == 3)
            {
                inputs.Add(NamedOnnxValue.CreateFromTensor(
                    name, new DenseTensor<float>(features, new[] { 1, numFrames, featDim })));
            }
            else if (lower.Contains("len"))
            {
                inputs.Add(NamedOnnxValue.CreateFromTensor(
                    name, new DenseTensor<int>(new[] { numFrames }, new[] { 1 })));
            }
            else if (lower.Contains("language") || lower == "lang")
            {
                inputs.Add(NamedOnnxValue.CreateFromTensor(
                    name, new DenseTensor<long>(new[] { _opts.LanguageId }, new[] { 1 })));
            }
            else if (lower.Contains("textnorm") || lower.Contains("text_norm") || lower.Contains("itn"))
            {
                inputs.Add(NamedOnnxValue.CreateFromTensor(
                    name, new DenseTensor<long>(new[] { _opts.TextNormId }, new[] { 1 })));
            }
        }

        using var results = _session.Run(inputs);

        foreach (var r in results)
        {
            var t = r.AsTensor<float>();
            if (t.Dimensions.Length == 3)
            {
                outFrames = t.Dimensions[1];
                vocab = t.Dimensions[2];
                return t.ToArray();
            }
        }

        throw new InvalidOperationException("NAR model produced no 3-D float logits output.");
    }

    public void Dispose() => _session.Dispose();
}
