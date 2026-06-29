using Stt.Abstractions.Ep;

namespace Stt.Core.Ep;

/// <summary>
/// Picks the best quantization variant of a model for a target execution provider (spec §9, §17,
/// Phase 2). NPU providers (QNN / VitisAI) require a fully static, quantized graph and prefer
/// <c>.int8</c>; DirectML prefers <c>.fp16</c> when present (else fp32); CPU uses fp32. Selection
/// honors the manifest's <c>providerSupport</c> — a model that omits an EP (e.g. dynamic-shape →
/// no "qnn") is never chosen for it.
/// </summary>
public static class ModelVariantSelector
{
    /// <summary>
    /// Given a model folder and the base file (e.g. <c>model.onnx</c>), return the variant file
    /// name best matching <paramref name="ep"/>, falling back to the base file when no preferred
    /// variant exists on disk.
    /// </summary>
    public static string SelectVariantFile(string folder, string baseFile, EpKind ep)
    {
        string stem = Path.GetFileNameWithoutExtension(baseFile);
        string ext = Path.GetExtension(baseFile); // .onnx

        string[] preference = ep switch
        {
            EpKind.Qnn or EpKind.VitisAI => new[] { "int8", "uint8" },
            EpKind.DirectML or EpKind.Cuda => new[] { "fp16" },
            EpKind.Cpu => new[] { "int8" },   // CPU: prefer int8 if shipped (~25% faster, lower CPU)
            _ => Array.Empty<string>(),
        };

        foreach (string q in preference)
        {
            string candidate = $"{stem}.{q}{ext}";
            if (File.Exists(Path.Combine(folder, candidate)))
                return candidate;
        }
        return baseFile;
    }

    /// <summary>Whether <paramref name="ep"/> is listed in the manifest's provider support.</summary>
    public static bool IsProviderSupported(IReadOnlyList<string> providerSupport, EpKind ep)
    {
        if (providerSupport.Count == 0) return ep == EpKind.Cpu; // unannotated ⇒ CPU-only safe default
        string name = ep switch
        {
            EpKind.Cpu => "cpu",
            EpKind.DirectML => "directml",
            EpKind.Cuda => "cuda",
            EpKind.Qnn => "qnn",
            EpKind.OpenVINO => "openvino",
            EpKind.VitisAI => "vitisai",
            _ => "cpu",
        };
        foreach (string p in providerSupport)
            if (string.Equals(p, name, StringComparison.OrdinalIgnoreCase)) return true;
        return false;
    }
}
