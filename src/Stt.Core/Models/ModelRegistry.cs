using Stt.Abstractions.Models;
using Stt.Abstractions.Pipeline;

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

    /// <summary>Infer a tentative manifest from file naming (spec §10.3): encoder/decoder/joiner ⇒ transducer; single model.onnx ⇒ ctc/nar.</summary>
    private static ModelManifest InferManifest(string folder)
    {
        bool Has(string name) => File.Exists(Path.Combine(folder, name));
        string? tokens = Has("tokens.txt") ? "tokens.txt" : null;
        string id = Path.GetFileName(folder.TrimEnd(Path.DirectorySeparatorChar));

        if (Has("encoder.onnx") && Has("decoder.onnx") && Has("joiner.onnx"))
        {
            return new ModelManifest
            {
                Id = id,
                DisplayName = id,
                Family = "transducer",
                DecoderType = "transducer",
                Runtime = new[] { "streaming" },
                Files = new ModelFiles { Encoder = "encoder.onnx", Decoder = "decoder.onnx", Joiner = "joiner.onnx", Tokens = tokens },
                Feature = new FeatureSpec { FrontEnd = "kaldi_fbank", Family = "KaldiFbankPovey", FeatureDim = 80 },
                Capabilities = new CapabilityFlags { StreamingCapable = true, OfflineCapable = false, Multilingual = true, NeedsVad = false },
                FolderPath = folder,
            };
        }

        string? single = FirstExisting(folder, "model.onnx", "model.int8.onnx", "ctc.onnx", "sense-voice.onnx", "sensevoice.onnx");
        if (single is not null)
        {
            bool senseVoice = single.Contains("sense", StringComparison.OrdinalIgnoreCase);
            return new ModelManifest
            {
                Id = id,
                DisplayName = id,
                Family = senseVoice ? "sense_voice" : "ctc",
                DecoderType = senseVoice ? "nar" : "ctc",
                Runtime = new[] { "offline" },
                Files = new ModelFiles { Model = single, Tokens = tokens },
                Feature = senseVoice
                    ? new FeatureSpec { FrontEnd = "kaldi_fbank", Family = "KaldiFbankLfrCmvn", FeatureDim = 560, Lfr = new[] { 7, 6 }, Cmvn = "metadata" }
                    : new FeatureSpec { FrontEnd = "kaldi_fbank", Family = "KaldiFbankPovey", FeatureDim = 80 },
                Capabilities = new CapabilityFlags { StreamingCapable = false, OfflineCapable = true, Multilingual = senseVoice, NeedsVad = true, NeedsLfrCmvn = senseVoice },
                FolderPath = folder,
            };
        }

        throw new InvalidOperationException(
            $"Folder '{folder}' has no manifest.json and no recognizable ONNX files (encoder/decoder/joiner or model.onnx).");
    }

    private static string? FirstExisting(string folder, params string[] names)
    {
        foreach (string n in names)
            if (File.Exists(Path.Combine(folder, n))) return n;
        return null;
    }
}
