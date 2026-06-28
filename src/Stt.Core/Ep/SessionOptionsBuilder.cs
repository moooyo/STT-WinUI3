using Microsoft.ML.OnnxRuntime;

namespace Stt.Core.Ep;

/// <summary>
/// Creates a configured ORT <see cref="SessionOptions"/> (spec §6: shared, core-limited intra-op
/// threading to avoid oversubscription; §9: graph optimization; EPContext compile cache wiring).
/// Phase 0 targets the CPU EP (the base ORT package); the structure is ready for DirectML/NPU EP
/// append in later phases and in the app's Windows ML selector.
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

        // Provider append for non-CPU devices is handled by ExecutionProviderSelector (and the
        // app's Windows ML selector), wrapped in try/catch with CPU fallback. CPU EP is implicit.
        return options;
    }

    private static void TrySetConfig(SessionOptions options, string key, string value)
    {
        try { options.AddSessionConfigEntry(key, value); }
        catch { /* older ORT or unsupported key — cache simply not enabled */ }
    }
}
