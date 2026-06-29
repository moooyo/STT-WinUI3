using Microsoft.ML.OnnxRuntime;
using Stt.Abstractions.Features;
using Stt.Abstractions.Models;
using Stt.Abstractions.Pipeline;
using Stt.Core.Features;

namespace Stt.Core.Models;

/// <summary>A validated pair of models for a pipeline mode (spec §10.3 ResolveCombination).</summary>
public sealed record ResolvedPipelineModels(
    ModelManifest? FirstPass,
    ModelManifest? SecondPass,
    PipelineMode Mode,
    bool RequiresVad);

/// <summary>
/// Catalog + sideload for user-supplied models (spec §10.3, D7). Scans the models root and extra
/// import paths; loads <c>manifest.json</c> when present, otherwise infers a tentative manifest
/// from file naming for the user to confirm. <see cref="ResolveCombination"/> enforces the legal
/// first/second-pass combinations from the capability flags (spec §11).
/// </summary>
public sealed class ModelRegistry : IModelRegistry
{
    private readonly Dictionary<string, ModelManifest> _models = new(StringComparer.OrdinalIgnoreCase);

    public ModelRegistry(string? modelsRoot = null, IEnumerable<string>? extraPaths = null)
    {
        if (!string.IsNullOrEmpty(modelsRoot) && Directory.Exists(modelsRoot))
            ScanRoot(modelsRoot);
        if (extraPaths is not null)
            foreach (string p in extraPaths)
                if (Directory.Exists(p)) TryImport(p);
    }

    private void ScanRoot(string root)
    {
        foreach (string dir in Directory.EnumerateDirectories(root))
            TryImport(dir);
        // A root that is itself a single model folder.
        if (Directory.GetFiles(root, "*.onnx").Length > 0 || File.Exists(Path.Combine(root, "manifest.json")))
            TryImport(root);
    }

    private void TryImport(string folder)
    {
        try { ImportFromFolder(folder); }
        catch { /* skip unreadable/non-model folders during scan */ }
    }

    public IReadOnlyList<ModelManifest> List() => _models.Values.ToList();

    public ModelManifest Get(string id) =>
        _models.TryGetValue(id, out var m) ? m : throw new KeyNotFoundException($"Model '{id}' not found.");

    public ModelManifest ImportFromFolder(string folderPath)
    {
        if (!Directory.Exists(folderPath))
            throw new DirectoryNotFoundException(folderPath);

        string manifestPath = Path.Combine(folderPath, "manifest.json");
        ModelManifest manifest = File.Exists(manifestPath)
            ? ModelManifestIo.Load(manifestPath) with { FolderPath = folderPath }
            : InferManifest(folderPath);

        if (string.IsNullOrEmpty(manifest.Id))
            manifest = manifest with { Id = Path.GetFileName(folderPath.TrimEnd(Path.DirectorySeparatorChar)) };

        _models[manifest.Id] = manifest;
        return manifest;
    }

    public void Remove(string id) => _models.Remove(id);

    /// <summary>
    /// Validate a first/second-pass model combination for a mode (spec §11): first pass must be
    /// streaming-capable; the offline pass must be offline-capable; two-pass and offline modes
    /// require a VAD. Throws <see cref="ArgumentException"/> with a human-readable reason.
    /// </summary>
    public ResolvedPipelineModels ResolveCombination(string? firstPassId, string? secondPassId, PipelineMode mode)
    {
        ModelManifest? p1 = firstPassId is null ? null : Get(firstPassId);
        ModelManifest? p2 = secondPassId is null ? null : Get(secondPassId);

        switch (mode)
        {
            case PipelineMode.OnePassStreaming:
                Require(p1 is not null, "OnePassStreaming requires a first-pass model.");
                RequireStreaming(p1!);
                return new ResolvedPipelineModels(p1, null, mode, RequiresVad: false);

            case PipelineMode.OnePassOffline:
                Require(p2 is not null, "OnePassOffline requires an offline (second-pass) model.");
                RequireOffline(p2!);
                return new ResolvedPipelineModels(null, p2, mode, RequiresVad: true);

            case PipelineMode.TwoPass:
                Require(p1 is not null, "TwoPass requires a streaming first-pass model.");
                Require(p2 is not null, "TwoPass requires an offline second-pass model.");
                RequireStreaming(p1!);
                RequireOffline(p2!);
                return new ResolvedPipelineModels(p1, p2, mode, RequiresVad: true);

            default:
                throw new ArgumentOutOfRangeException(nameof(mode));
        }
    }

    private static void RequireStreaming(ModelManifest m)
    {
        if (!m.Capabilities.StreamingCapable)
            throw new ArgumentException(
                $"Model '{m.Id}' ({m.Family}) is not streaming-capable and cannot be the first pass " +
                "(e.g. Whisper/SenseVoice are offline-only).");
    }

    private static void RequireOffline(ModelManifest m)
    {
        if (!m.Capabilities.OfflineCapable)
            throw new ArgumentException(
                $"Model '{m.Id}' ({m.Family}) is not offline-capable and cannot be the second pass.");
    }

    private static void Require(bool condition, string message)
    {
        if (!condition) throw new ArgumentException(message);
    }

    /// <summary>
    /// Infer a tentative manifest from a model folder (spec §10.3). File discovery is glob-based
    /// (handles sherpa's prefixed names like <c>tiny.en-encoder.onnx</c>): encoder+decoder+joiner ⇒
    /// transducer; encoder+decoder (no joiner) ⇒ Whisper (Family C); a single graph ⇒ probed via ONNX
    /// metadata so the feature family is correct (NeMo→D, SenseVoice→B, else CTC→A).
    /// </summary>
    private static ModelManifest InferManifest(string folder)
    {
        string id = Path.GetFileName(folder.TrimEnd(Path.DirectorySeparatorChar));
        string? Find(string suffix) => Directory.GetFiles(folder, "*" + suffix)
            .Select(Path.GetFileName).Where(n => n is not null && !n.Contains(".int8.")).FirstOrDefault();
        string? tokens = Find("tokens.txt");

        string? enc = Find("encoder.onnx");
        string? dec = Find("decoder.onnx");
        string? joi = Find("joiner.onnx");

        // Whisper (Family C): encoder + decoder, NO joiner.
        if (enc is not null && dec is not null && joi is null)
        {
            ModelProbe? probe = TryProbe(Path.Combine(folder, enc));
            // n_mels is the Whisper arbiter: prefer explicit metadata, else the encoder's mel axis
            // ([N, n_mels, 3000] → FeatureDim), so large-v3 (128) auto-detects instead of forcing 80.
            int nMels = probe?.NMels ?? (probe is { FeatureDim: > 0 } ? probe.FeatureDim : 80);
            bool multi = probe is not null && probe.Metadata.TryGetValue("is_multilingual", out var im) && im != "0";
            return new ModelManifest
            {
                Id = id,
                DisplayName = id,
                Family = "whisper",
                DecoderType = "ar",
                Runtime = new[] { "offline" },
                Files = new ModelFiles { Encoder = enc, Decoder = dec, Tokens = tokens },
                Feature = new FeatureSpec { FrontEnd = "whisper", Family = "WhisperLogMel", FeatureDim = nMels },
                Capabilities = new CapabilityFlags { OfflineCapable = true, Multilingual = multi, NeedsVad = true },
                Languages = multi ? new[] { "multilingual" } : new[] { "en" },
                License = "MIT",
                FolderPath = folder,
            };
        }

        // Transducer (Family A): encoder + decoder + joiner.
        if (enc is not null && dec is not null && joi is not null)
        {
            return new ModelManifest
            {
                Id = id,
                DisplayName = id,
                Family = "transducer",
                DecoderType = "transducer",
                Runtime = new[] { "streaming" },
                Files = new ModelFiles { Encoder = enc, Decoder = dec, Joiner = joi, Tokens = tokens },
                Feature = new FeatureSpec { FrontEnd = "kaldi_fbank", Family = "KaldiFbankPovey", FeatureDim = 80 },
                Capabilities = new CapabilityFlags { StreamingCapable = true, OfflineCapable = false, Multilingual = true, NeedsVad = false },
                FolderPath = folder,
            };
        }

        // Single graph: probe ONNX metadata so the feature family is correct (NeMo vs SenseVoice vs CTC).
        string? single = FirstExisting(folder, "model.onnx", "ctc.onnx", "sense-voice.onnx", "sensevoice.onnx")
            ?? Find(".onnx");
        if (single is not null)
            return InferSingleGraph(folder, id, single, tokens);

        throw new InvalidOperationException(
            $"Folder '{folder}' has no manifest.json and no recognizable ONNX files (encoder/decoder[/joiner] or a single model graph).");
    }

    private static ModelManifest InferSingleGraph(string folder, string id, string single, string? tokens)
    {
        ModelProbe? probe = TryProbe(Path.Combine(folder, single));
        AsrFeatureFamily fam = probe is not null ? FeatureFamilyDetector.Detect(probe) : AsrFeatureFamily.Auto;
        int dim = probe?.FeatureDim ?? 0;

        // Name-based fallback when the metadata can't classify (unreadable graph / ambiguous export):
        // a "sense"-named single graph is SenseVoice (Family B).
        if (fam == AsrFeatureFamily.Auto && single.Contains("sense", StringComparison.OrdinalIgnoreCase))
            fam = AsrFeatureFamily.KaldiFbankLfrCmvn;

        return fam switch
        {
            AsrFeatureFamily.NemoMel => new ModelManifest
            {
                Id = id, DisplayName = id, Family = "nemo", DecoderType = "ctc", Runtime = new[] { "offline" },
                Files = new ModelFiles { Model = single, Tokens = tokens },
                Feature = new FeatureSpec { FrontEnd = "nemo", Family = "NemoMel", FeatureDim = dim > 0 ? dim : 80 },
                Capabilities = new CapabilityFlags { OfflineCapable = true, Multilingual = false, NeedsVad = true },
                FolderPath = folder,
            },
            AsrFeatureFamily.KaldiFbankLfrCmvn => new ModelManifest
            {
                Id = id, DisplayName = id, Family = "sense_voice", DecoderType = "nar", Runtime = new[] { "offline" },
                Files = new ModelFiles { Model = single, Tokens = tokens },
                Feature = new FeatureSpec { FrontEnd = "kaldi_fbank", Family = "KaldiFbankLfrCmvn", FeatureDim = dim > 0 ? dim : 560, Lfr = new[] { 7, 6 }, Cmvn = "metadata" },
                Capabilities = new CapabilityFlags { OfflineCapable = true, Multilingual = true, NeedsVad = true, NeedsLfrCmvn = true },
                FolderPath = folder,
            },
            _ => new ModelManifest   // KaldiFbankPovey / Auto → plain CTC (Family A)
            {
                Id = id, DisplayName = id, Family = "ctc", DecoderType = "ctc", Runtime = new[] { "offline" },
                Files = new ModelFiles { Model = single, Tokens = tokens },
                Feature = new FeatureSpec { FrontEnd = "kaldi_fbank", Family = "KaldiFbankPovey", FeatureDim = dim > 0 ? dim : 80 },
                Capabilities = new CapabilityFlags { OfflineCapable = true, NeedsVad = true },
                FolderPath = folder,
            },
        };
    }

    private static ModelProbe? TryProbe(string onnxPath)
    {
        try
        {
            using var session = new InferenceSession(onnxPath);
            return ModelMetadataReader.FromSession(session);
        }
        catch { return null; }   // unreadable / not loadable — fall back to naming heuristics
    }

    private static string? FirstExisting(string folder, params string[] names)
    {
        foreach (string n in names)
            if (File.Exists(Path.Combine(folder, n))) return n;
        return null;
    }
}
