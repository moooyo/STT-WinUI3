using Stt.Abstractions.Ep;

namespace Stt.Core.Ep;

/// <summary>The device chosen for a session plus whether it was a CPU fallback (spec §9, §14).</summary>
public sealed record EpResolution(EpDeviceInfo Device, bool FellBackToCpu);

/// <summary>
/// Pure execution-provider selection logic (spec §9), separated from ORT so it is unit testable.
/// Given a preference and the freshly enumerated device list, it picks the matching device, or —
/// when the preferred device is absent and fallback is allowed — degrades to CPU. PREFER_* style
/// preferences inherently fall back. This is the decision; <see cref="ExecutionProviderSelector"/>
/// applies it to a real <c>SessionOptions</c>.
/// </summary>
public static class EpResolver
{
    public static EpResolution Resolve(EpPreference pref, IReadOnlyList<EpDeviceInfo> devices)
    {
        // CPU is always available even if the enumeration omitted it.
        EpDeviceInfo cpu = FindKind(devices, EpKind.Cpu) ?? EpDeviceInfo.Cpu;

        if (pref.Kind == EpKind.Cpu)
            return new EpResolution(cpu, FellBackToCpu: false);

        EpDeviceInfo? match = FindKind(devices, pref.Kind);
        if (match is not null)
            return new EpResolution(match, FellBackToCpu: false);

        if (pref.AllowFallbackToCpu)
            return new EpResolution(cpu, FellBackToCpu: true);

        throw new InvalidOperationException(
            $"Execution provider {pref.Kind} is unavailable and CPU fallback is disabled.");
    }

    private static EpDeviceInfo? FindKind(IReadOnlyList<EpDeviceInfo> devices, EpKind kind)
    {
        foreach (var d in devices)
            if (d.Kind == kind) return d;
        return null;
    }
}
