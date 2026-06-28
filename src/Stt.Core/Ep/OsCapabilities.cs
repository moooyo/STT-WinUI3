namespace Stt.Core.Ep;

/// <summary>
/// Runtime capability gates for execution providers (spec §4, §9): CPU + DirectML are available on
/// Win10 1809+; NPU and optimized GPU EPs require Win11 24H2 (build 26100)+. Code paths consult
/// these gates and fall back to DirectML/CPU below the threshold.
/// </summary>
public static class OsCapabilities
{
    /// <summary>Windows 11 24H2 build number — the floor for NPU / optimized EPs.</summary>
    public const int Build24H2 = 26100;

    /// <summary>Windows 10 1809 build number — the floor for CPU + DirectML.</summary>
    public const int Build1809 = 17763;

    public static int CurrentBuild => Environment.OSVersion.Version.Build;

    /// <summary>True when NPU / optimized GPU EPs may be attempted (Win11 24H2+).</summary>
    public static bool SupportsNpuEps => OperatingSystem.IsWindows() && CurrentBuild >= Build24H2;

    /// <summary>True when DirectML / CPU EPs are available (Win10 1809+).</summary>
    public static bool SupportsDirectML => OperatingSystem.IsWindows() && CurrentBuild >= Build1809;

    /// <summary>
    /// Gate an EP build by OS: returns true when <paramref name="build"/> requirement is met.
    /// Pure (does not read the live OS) so it is unit testable.
    /// </summary>
    public static bool MeetsBuild(int currentBuild, int requiredBuild) => currentBuild >= requiredBuild;
}
