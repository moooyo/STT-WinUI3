namespace Stt.Core.Ep;

/// <summary>Records why the last EP session creation fell back to CPU, surfaced to the UI for diagnosis.</summary>
public static class EpDiagnostics
{
    public static string? LastFallbackReason { get; set; }
}
