using Stt.Abstractions.Ep;

namespace Stt.Core.Ep;

/// <summary>Hardware class behind an execution provider (spec §9: filter by EpName + HardwareDevice.Type).</summary>
public enum HardwareKind { Cpu, Gpu, Npu }

/// <summary>
/// A discoverable execution-provider device, mirroring ORT's <c>OrtEpDevice</c> (EP name +
/// hardware type + version + driver). Enumerated fresh per session build (spec §9: don't cache
/// the device list).
/// </summary>
public sealed record EpDeviceInfo(
    string EpName,
    HardwareKind Hardware,
    EpKind Kind,
    string EpVersion = "",
    string Driver = "")
{
    public static EpDeviceInfo Cpu { get; } = new("CPUExecutionProvider", HardwareKind.Cpu, EpKind.Cpu);
}
