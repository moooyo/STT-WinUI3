using System.Linq;
using System.Reflection;
using Stt.Abstractions.Decoders;
using Stt.Core.Ep;

namespace Stt.Core.Tests.Abstractions;

/// <summary>
/// Executable guard for the spec §8.5 / §17 invariant: the optional Whisper-genai plugin (and the
/// onnxruntime-genai native stack it carries) must NOT leak into the core engine or the unified
/// Windows ML EP path. The plugin references Core — never the reverse — so neither Stt.Core nor
/// Stt.Abstractions may name the plugin or Microsoft.ML.OnnxRuntimeGenAI among their references.
/// </summary>
public class PluginIsolationTests
{
    private static readonly string[] Forbidden =
        { "Stt.Plugins.WhisperGenAi", "Microsoft.ML.OnnxRuntimeGenAI" };

    [Fact]
    public void Core_Does_Not_Reference_WhisperGenAi_Or_Genai()
    {
        AssertNoForbiddenReferences(typeof(ExecutionProviderSelector).Assembly);
    }

    [Fact]
    public void Abstractions_Does_Not_Reference_WhisperGenAi_Or_Genai()
    {
        AssertNoForbiddenReferences(typeof(IAsrDecoder).Assembly);
    }

    private static void AssertNoForbiddenReferences(Assembly assembly)
    {
        var referenced = assembly.GetReferencedAssemblies().Select(a => a.Name).ToHashSet();
        foreach (string forbidden in Forbidden)
            Assert.DoesNotContain(forbidden, referenced);
    }
}
