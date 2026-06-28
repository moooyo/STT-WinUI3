namespace Stt.Abstractions.Ep;

/// <summary>
/// A user/UI preference for which execution provider to use. <see cref="AllowFallbackToCpu"/>
/// lets the selector degrade to CPU when the requested device is unavailable or session
/// creation fails (spec §9, §14).
/// </summary>
/// <param name="Kind">Preferred provider.</param>
/// <param name="AllowFallbackToCpu">When true, the selector silently falls back to CPU.</param>
public sealed record EpPreference(EpKind Kind, bool AllowFallbackToCpu = true)
{
    /// <summary>"Auto": let the selector pick the most efficient available provider.</summary>
    public static EpPreference Auto { get; } = new(EpKind.Cpu, AllowFallbackToCpu: true);
}
