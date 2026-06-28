using Microsoft.ML.OnnxRuntime;
using Stt.Abstractions.Models;
using Stt.Core.Text;

namespace Stt.Core.Models;

/// <summary>Outcome of loading + validating a model's primary graph (spec §10.2).</summary>
public sealed record ModelLoadResult(ModelProbe Probe, ValidationReport Report);

/// <summary>
/// Opens a model's primary ONNX graph, reads its metadata + input dims, and runs the five-layer
/// validation (spec §10.2). Gate 5 (built-in WAV self-test) is optional and only attempted when a
/// front-end + decoder are wired; here we cover gates 1–4 plus the structural session open.
/// Requires the ONNX file + ORT native runtime, so it is exercised by skippable integration tests.
/// </summary>
public static class ModelLoader
{
    /// <summary>
    /// Validate the model whose primary graph is <paramref name="onnxPath"/> against
    /// <paramref name="manifest"/>. <paramref name="tokensPath"/> (if provided) enables gate 4.
    /// </summary>
    public static ModelLoadResult LoadAndValidate(string onnxPath, ModelManifest manifest, string? tokensPath = null, SessionOptions? sessionOptions = null)
    {
        if (!File.Exists(onnxPath)) throw new FileNotFoundException("Model graph not found.", onnxPath);

        using var session = sessionOptions is null ? new InferenceSession(onnxPath) : new InferenceSession(onnxPath, sessionOptions);
        ModelProbe probe = ModelMetadataReader.FromSession(session);

        TokenTable? tokens = null;
        if (!string.IsNullOrEmpty(tokensPath) && File.Exists(tokensPath))
            tokens = TokenTable.Load(tokensPath);

        ValidationReport report = ModelValidation.Validate(manifest, probe, tokens);
        return new ModelLoadResult(probe, report);
    }
}
