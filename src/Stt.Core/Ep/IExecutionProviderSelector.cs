using Microsoft.ML.OnnxRuntime;
using Stt.Abstractions.Ep;

namespace Stt.Core.Ep;

/// <summary>
/// Builds an ORT <see cref="SessionOptions"/> with a selected execution provider, handling
/// compile-cache wiring and CPU fallback (spec §5.1, §9). The selector returns ORT's
/// <c>SessionOptions</c>, which is why this interface lives in Core (it cannot sit in the
/// dependency-free Abstractions layer).
/// </summary>
public interface IExecutionProviderSelector
{
    /// <summary>
    /// Build session options for a model identified by <paramref name="modelHash"/> using the
    /// given preference. Internally resolves the device, wires the EPContext compile cache, and
    /// degrades to CPU on unavailability or session-creation failure.
    /// </summary>
    SessionOptions BuildSessionOptions(EpPreference pref, string modelHash);

    /// <summary>The device chosen by the most recent <see cref="BuildSessionOptions"/> call.</summary>
    EpResolution? LastResolution { get; }
}
