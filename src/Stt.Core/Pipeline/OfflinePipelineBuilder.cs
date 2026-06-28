using Microsoft.ML.OnnxRuntime;
using Stt.Abstractions.Decoders;
using Stt.Abstractions.Ep;
using Stt.Abstractions.Features;
using Stt.Abstractions.Models;
using Stt.Abstractions.Vad;
using Stt.Core.Decoders;
using Stt.Core.Ep;
using Stt.Core.Features;
using Stt.Core.Text;
using Stt.Core.Vad;

namespace Stt.Core.Pipeline;

/// <summary>Components of an offline (second-pass / OnePassOffline) decode chain.</summary>
public sealed record OfflineChain(IVad Vad, IFeatureFrontend Frontend, IAsrDecoder Decoder) : IDisposable
{
    public void Dispose()
    {
        Vad.Dispose();
        (Frontend as IDisposable)?.Dispose();
        Decoder.Dispose();
    }
}

/// <summary>
/// Wires an offline SenseVoice/Paraformer decode chain from a model folder (spec §8.4, §10): loads
/// the ONNX with the selected EP, reads CMVN stats from metadata (fail-loud if missing for Family
/// B), builds the kaldi-fbank front-end + NAR decoder, and a Silero VAD. Used by the app's
/// OnePassOffline pipeline (Phase 0). Requires the real model files + ORT native runtime.
/// </summary>
public static class OfflinePipelineBuilder
{
    public static OfflineChain BuildOffline(
        ModelManifest manifest,
        string vadModelPath,
        IExecutionProviderSelector epSelector,
        EpPreference epPreference)
    {
        if (manifest.FolderPath is null)
            throw new InvalidOperationException("Manifest has no folder path; import it via the registry first.");

        string modelFile = manifest.Files.Model
            ?? throw new InvalidOperationException($"Offline model '{manifest.Id}' has no 'model' file in its manifest.");
        // Pick the quantization variant best matching the EP (spec §9, Phase 2).
        modelFile = ModelVariantSelector.SelectVariantFile(manifest.FolderPath, modelFile, epPreference.Kind);
        string modelPath = Path.Combine(manifest.FolderPath, modelFile);
        string tokensPath = Path.Combine(manifest.FolderPath, manifest.Files.Tokens
            ?? throw new InvalidOperationException("Offline model requires a tokens file."));

        string modelHash = ComputeShortHash(modelPath);
        SessionOptions opts = epSelector.BuildSessionOptions(epPreference, modelHash);
        var session = new InferenceSession(modelPath, opts);

        // Build the front-end (Family B requires CMVN from metadata — fail loud if absent).
        IFeatureFrontend frontend = BuildFrontend(manifest, session);

        var tokens = TokenTable.Load(tokensPath);
        var runner = new OrtNarModelRunner(session);
        var decoder = new NarDecoder(runner, tokens, new NarDecoderOptions
        {
            QuerySlots = manifest.Family.Contains("sense", StringComparison.OrdinalIgnoreCase) ? 4 : 0,
            Blank = DetectBlankId(tokens),
        });

        var vad = new SileroVad(vadModelPath, new VadOptions());
        return new OfflineChain(vad, frontend, decoder);
    }

    private static IFeatureFrontend BuildFrontend(ModelManifest manifest, InferenceSession session)
    {
        AsrFeatureFamily fam = Enum.TryParse<AsrFeatureFamily>(manifest.Feature.Family, true, out var f)
            ? f : AsrFeatureFamily.Auto;

        if (fam == AsrFeatureFamily.KaldiFbankLfrCmvn)
        {
            var meta = session.ModelMetadata.CustomMetadataMap;
            float[] negMean = ParseFloatArray(meta, "neg_mean", "cmvn_neg_mean")
                ?? throw new InvalidOperationException("Family B model is missing CMVN 'neg_mean' in metadata (spec §10.2).");
            float[] invStddev = ParseFloatArray(meta, "inv_stddev", "cmvn_inv_stddev")
                ?? throw new InvalidOperationException("Family B model is missing CMVN 'inv_stddev' in metadata (spec §10.2).");

            int[] lfr = manifest.Feature.Lfr ?? new[] { 7, 6 };
            int numBins = manifest.Feature.FeatureDim / lfr[0];
            return new KaldiFbankFrontend(new FbankOptions
            {
                NumBins = numBins,
                WindowType = "hamming",
                Lfr = (lfr[0], lfr[1]),
                CmvnNegMean = negMean,
                CmvnInvStddev = invStddev,
                NormalizeSamples = true,
            });
        }

        // Family C — Whisper / Qwen log-mel (80 or 128 bins).
        if (fam == AsrFeatureFamily.WhisperLogMel)
            return new WhisperMelFrontend(manifest.Feature.FeatureDim);

        // Family D — NeMo / GigaAM librosa-mel + per-feature norm.
        if (fam == AsrFeatureFamily.NemoMel)
            return new NemoMelFrontend(manifest.Feature.FeatureDim);

        // Family A default.
        return new KaldiFbankFrontend(FbankOptions.FamilyA(manifest.Feature.FeatureDim));
    }

    private static float[]? ParseFloatArray(IReadOnlyDictionary<string, string> meta, params string[] keys)
    {
        foreach (string k in keys)
        {
            if (meta.TryGetValue(k, out var v) && !string.IsNullOrWhiteSpace(v))
            {
                string[] parts = v.Split(new[] { ',', ' ' }, StringSplitOptions.RemoveEmptyEntries);
                var arr = new float[parts.Length];
                for (int i = 0; i < parts.Length; i++) arr[i] = float.Parse(parts[i]);
                return arr;
            }
        }
        return null;
    }

    private static string ComputeShortHash(string path)
    {
        var fi = new FileInfo(path);
        return $"{Path.GetFileNameWithoutExtension(path)}-{fi.Length}";
    }

    /// <summary>
    /// Resolve the CTC blank id from the token table: NeMo uses a trailing <c>&lt;blk&gt;</c> (id ==
    /// vocab_size), FunASR/SenseVoice use id 0. Falls back to 0 when no blank token is named.
    /// </summary>
    private static int DetectBlankId(TokenTable tokens)
    {
        foreach (string p in new[] { "<blk>", "<blank>", "<pad>", "<eps>" })
            if (tokens.TryId(p, out int id)) return id;
        return 0;
    }
}
