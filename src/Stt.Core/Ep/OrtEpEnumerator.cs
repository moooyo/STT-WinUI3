using Microsoft.ML.OnnxRuntime;
using Stt.Abstractions.Ep;

namespace Stt.Core.Ep;

/// <summary>
/// Enumerates execution-provider devices from ORT's autoEP catalog (<c>OrtEnv.GetEpDevices()</c>,
/// spec §9). In the base CPU-only package this returns just the CPU EP; once the app registers the
/// Windows ML certified EPs (<c>ExecutionProviderCatalog.EnsureAndRegisterCertifiedAsync</c>),
/// DirectML and vendor NPU devices appear here too. Mapped to <see cref="EpDeviceInfo"/> with the
/// backing <c>OrtEpDevice</c> retained for autoEP append. Fully defensive: any failure degrades to
/// the synthetic CPU device so headless engine tests (no native EPs) keep working.
/// </summary>
public static class OrtEpEnumerator
{
    /// <summary>Fresh device list (never cache — spec §9). Always includes CPU.</summary>
    public static IReadOnlyList<EpDeviceInfo> Enumerate()
    {
        var devices = new List<EpDeviceInfo>();
        try
        {
            foreach (OrtEpDevice d in OrtEnv.Instance().GetEpDevices())
            {
                EpKind kind = ClassifyEp(d.EpName);
                HardwareKind hw = ClassifyHardware(d, kind);
                string ver = TryMeta(d, "version");
                string driver = TryMeta(d, "driver", "driver_version");
                devices.Add(new EpDeviceInfo(d.EpName, hw, kind, ver, driver) { Native = d });
            }
        }
        catch { /* older/CPU-only native or no autoEP — fall through to CPU default */ }

        if (!devices.Exists(x => x.Kind == EpKind.Cpu))
            devices.Add(EpDeviceInfo.Cpu);
        return devices;
    }

    private static EpKind ClassifyEp(string epName)
    {
        string n = epName.ToLowerInvariant();
        if (n.Contains("dml") || n.Contains("directml")) return EpKind.DirectML;
        if (n.Contains("qnn")) return EpKind.Qnn;
        if (n.Contains("openvino")) return EpKind.OpenVINO;
        if (n.Contains("vitis")) return EpKind.VitisAI;
        if (n.Contains("tensorrt") || n.Contains("cuda") || n.Contains("nvidia")) return EpKind.Cuda;
        return EpKind.Cpu;
    }

    private static HardwareKind ClassifyHardware(OrtEpDevice d, EpKind kind)
    {
        try
        {
            return d.HardwareDevice.Type switch
            {
                OrtHardwareDeviceType.GPU => HardwareKind.Gpu,
                OrtHardwareDeviceType.NPU => HardwareKind.Npu,
                _ => HardwareKind.Cpu,
            };
        }
        catch
        {
            // Infer from the EP when the hardware-device probe is unavailable.
            return kind is EpKind.Qnn or EpKind.VitisAI ? HardwareKind.Npu
                 : kind is EpKind.DirectML or EpKind.Cuda or EpKind.OpenVINO ? HardwareKind.Gpu
                 : HardwareKind.Cpu;
        }
    }

    private static string TryMeta(OrtEpDevice d, params string[] keys)
    {
        try
        {
            var meta = d.EpMetadata.Entries;
            foreach (string k in keys)
                if (meta.TryGetValue(k, out var v) && !string.IsNullOrWhiteSpace(v)) return v;
        }
        catch { }
        return "";
    }
}
