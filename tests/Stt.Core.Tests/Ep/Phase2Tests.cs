using Microsoft.ML.OnnxRuntime;
using Stt.Abstractions.Ep;
using Stt.Core.Ep;

namespace Stt.Core.Tests.Ep;

public class ModelVariantSelectorTests
{
    private static string MakeFolder(params string[] files)
    {
        string dir = Path.Combine(Path.GetTempPath(), $"var_{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        foreach (string f in files) File.WriteAllText(Path.Combine(dir, f), "");
        return dir;
    }

    [Fact]
    public void Npu_Prefers_Int8_When_Present()
    {
        string dir = MakeFolder("model.onnx", "model.int8.onnx");
        try { Assert.Equal("model.int8.onnx", ModelVariantSelector.SelectVariantFile(dir, "model.onnx", EpKind.Qnn)); }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public void DirectML_Prefers_Fp16_Else_Base()
    {
        string dir = MakeFolder("model.onnx");
        try
        {
            Assert.Equal("model.onnx", ModelVariantSelector.SelectVariantFile(dir, "model.onnx", EpKind.DirectML));
            File.WriteAllText(Path.Combine(dir, "model.fp16.onnx"), "");
            Assert.Equal("model.fp16.onnx", ModelVariantSelector.SelectVariantFile(dir, "model.onnx", EpKind.DirectML));
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public void Cpu_Uses_Base_File()
    {
        string dir = MakeFolder("model.onnx", "model.int8.onnx");
        try { Assert.Equal("model.onnx", ModelVariantSelector.SelectVariantFile(dir, "model.onnx", EpKind.Cpu)); }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public void ProviderSupport_Gates_Selection()
    {
        Assert.True(ModelVariantSelector.IsProviderSupported(new[] { "cpu", "directml" }, EpKind.DirectML));
        Assert.False(ModelVariantSelector.IsProviderSupported(new[] { "cpu", "directml" }, EpKind.Qnn));
        // Unannotated → CPU-only safe default.
        Assert.True(ModelVariantSelector.IsProviderSupported(Array.Empty<string>(), EpKind.Cpu));
        Assert.False(ModelVariantSelector.IsProviderSupported(Array.Empty<string>(), EpKind.DirectML));
    }
}

public class OsCapabilitiesTests
{
    [Fact]
    public void MeetsBuild_Gates_By_Number()
    {
        Assert.True(OsCapabilities.MeetsBuild(26100, OsCapabilities.Build24H2));
        Assert.True(OsCapabilities.MeetsBuild(27000, OsCapabilities.Build24H2));
        Assert.False(OsCapabilities.MeetsBuild(22631, OsCapabilities.Build24H2)); // Win11 23H2 < 24H2
        Assert.True(OsCapabilities.MeetsBuild(17763, OsCapabilities.Build1809));
    }
}

public class FixedShapeTests
{
    [Fact]
    public void ApplyFixedShapes_Does_Not_Throw_On_Unknown_Dims()
    {
        using var options = new SessionOptions();
        // No model loaded; overriding free dims is a no-op-safe operation.
        ModelVariantSelectorFixedShapeHelper(options);
    }

    private static void ModelVariantSelectorFixedShapeHelper(SessionOptions options)
    {
        SessionOptionsBuilder.ApplyFixedShapes(options, new Dictionary<string, int> { ["T"] = 39, ["N"] = 1 });
    }
}
