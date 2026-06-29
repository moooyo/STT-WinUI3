using Microsoft.ML.OnnxRuntime;
using Stt.Abstractions.Ep;

namespace Stt.Core.Ep;

/// <summary>
/// Default <see cref="IExecutionProviderSelector"/> (spec §9). Enumerates devices fresh per build
/// (injectable for tests / Windows ML override), resolves the preference via <see cref="EpResolver"/>,
/// wires the EPContext compile cache, and degrades to CPU on unavailability. The base ORT package
/// exposes CPU only; the app substitutes a Windows ML-backed enumerator to surface DirectML/NPU.
/// </summary>
public sealed class ExecutionProviderSelector : IExecutionProviderSelector
{
    private readonly Func<IReadOnlyList<EpDeviceInfo>> _enumerate;
    private readonly CompiledModelCache? _cache;
    private readonly int _intraOpThreads;

    public ExecutionProviderSelector(
        Func<IReadOnlyList<EpDeviceInfo>>? enumerateDevices = null,
        CompiledModelCache? cache = null,
        int intraOpThreads = 0)
    {
        // Default enumeration: ORT's autoEP catalog (CPU only until the app registers Windows ML EPs).
        _enumerate = enumerateDevices ?? OrtEpEnumerator.Enumerate;
        _cache = cache;
        _intraOpThreads = intraOpThreads;
    }

    public EpResolution? LastResolution { get; private set; }

    public SessionOptions BuildSessionOptions(EpPreference pref, string modelHash)
    {
        IReadOnlyList<EpDeviceInfo> devices = _enumerate();   // fresh per build — never cached
        EpResolution resolution = EpResolver.Resolve(pref, devices);
        LastResolution = resolution;

        string? cachePath = null;
        if (_cache is not null && resolution.Device.Kind != EpKind.Cpu)
        {
            cachePath = _cache.ContextPath(
                modelHash, resolution.Device.EpName, resolution.Device.EpVersion, resolution.Device.Driver);
        }

        return SessionOptionsBuilder.Build(resolution, _intraOpThreads, cachePath);
    }
}
