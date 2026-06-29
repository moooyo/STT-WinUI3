using Stt.Abstractions.Ep;

namespace Stt.Core.Ep;

/// <summary>Hardware class behind an execution provider (spec §9: filter by EpName + HardwareDevice.Type).</summary>
public enum HardwareKind { Cpu, Gpu, Npu }

/// <summary>
/// A discoverable execution-provider device, mirroring ORT's <c>OrtEpDevice</c> (EP name +
/// hardware type + version + driver). Enumerated fresh per session build (spec §9: don't cache
/// the device list). <see cref="Native"/> carries the underlying ORT EP device so the session
/// builder can append it via the autoEP API; it is null for the synthetic CPU default and in
/// headless tests.
/// </summary>
public sealed record EpDeviceInfo(
    string EpName,
    HardwareKind Hardware,
    EpKind Kind,
    string EpVersion = "",
    string Driver = "")
{
    /// <summary>The backing <c>OrtEpDevice</c> for autoEP append, or null (CPU default / tests).</summary>
    public object? Native { get; init; }

    public static EpDeviceInfo Cpu { get; } = new("CPUExecutionProvider", HardwareKind.Cpu, EpKind.Cpu);
}
