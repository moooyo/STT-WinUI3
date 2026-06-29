using Microsoft.ML.OnnxRuntime;
using Stt.Abstractions.Ep;

namespace Stt.Core.Ep;

/// <summary>
/// Creates a configured ORT <see cref="SessionOptions"/> (spec §6: shared, core-limited intra-op
/// threading to avoid oversubscription; §9: graph optimization; EPContext compile cache wiring).
/// Appends the resolved non-CPU execution provider (DirectML / NPU) through ORT's autoEP API when a
/// discovered <c>OrtEpDevice</c> is supplied; CPU is implicit and any unappendable device falls back
/// to CPU. The app populates real devices by registering the Windows ML EP catalog.
/// </summary>
public static class SessionOptionsBuilder
{
    /// <summary>
    /// Build options for the resolved device. <paramref name="intraOpThreads"/> caps the intra-op
    /// pool (0 = ORT default). <paramref name="compileCachePath"/>, when non-null, enables the
    /// EPContext cache at that path.
    /// </summary>
    public static SessionOptions Build(EpResolution resolution, int intraOpThreads = 0, string? compileCachePath = null)
    {
        var options = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
        };

        if (intraOpThreads > 0)
            options.IntraOpNumThreads = intraOpThreads;

        if (!string.IsNullOrEmpty(compileCachePath))
        {
            // EPContext: cache the EP-compiled graph next to the model (spec §9). Defensive —
            // unsupported on some providers/versions.
            TrySetConfig(options, "ep.context_enable", "1");
            TrySetConfig(options, "ep.context_file_path", compileCachePath!);
            TrySetConfig(options, "ep.context_embed_mode", "0");
        }

        // Append the resolved non-CPU EP. Prefer the discovered autoEP device; for DirectML (always
        // built into Windows ML) fall back to the direct DML API so GPU is used even when autoEP
        // didn't enumerate a device. CPU is implicit; any failure degrades to CPU (FellBackToCpu).
        if (resolution.Device.Kind != EpKind.Cpu)
        {
            bool gated = resolution.Device.Kind == EpKind.DirectML
                ? OsCapabilities.SupportsDirectML
                : OsCapabilities.SupportsNpuEps;
            if (gated)
            {
                if (resolution.Device.Native is OrtEpDevice ep) TryAppendEp(options, ep);
                else if (resolution.Device.Kind == EpKind.DirectML)
                    EpDiagnostics.LastFallbackReason = "DirectML EP not registered in Windows ML on this machine — use CUDA (NVIDIA) or run Download; CPU in use";
            }
        }

        return options;
    }

    /// <summary>Append a single discovered EP device; swallow so an unsupported device falls back to CPU.</summary>
    private static void TryAppendEp(SessionOptions options, OrtEpDevice ep)
    {
        try { options.AppendExecutionProvider(OrtEnv.Instance(), new[] { ep }, new Dictionary<string, string>()); }
        catch (Exception e) { EpDiagnostics.LastFallbackReason = "append device: " + e.Message; }
    }

    private static void TrySetConfig(SessionOptions options, string key, string value)
    {
        try { options.AddSessionConfigEntry(key, value); }
        catch { /* older ORT or unsupported key — cache simply not enabled */ }
    }

    /// <summary>
    /// Pin dynamic axes to fixed sizes (spec §9 iron rule: DirectML dynamic axes are ~5× slower).
    /// Maps free-dimension names (e.g. <c>"T"</c>, <c>"N"</c>) to concrete values via ORT's
    /// <c>AddFreeDimensionOverrideByName</c>. Apply before session creation.
    /// </summary>
    public static void ApplyFixedShapes(SessionOptions options, IReadOnlyDictionary<string, int> overrides)
    {
        foreach (var kv in overrides)
        {
            try { options.AddFreeDimensionOverrideByName(kv.Key, kv.Value); }
            catch { /* dim name not present in this model — ignore */ }
        }
    }
}
