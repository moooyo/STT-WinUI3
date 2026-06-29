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

    [Fact]
    public void VitisAI_Prefers_Int8_Then_Uint8()
    {
        string dir = MakeFolder("model.onnx", "model.int8.onnx");
        try { Assert.Equal("model.int8.onnx", ModelVariantSelector.SelectVariantFile(dir, "model.onnx", EpKind.VitisAI)); }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public void Npu_Falls_Back_To_Uint8_When_Int8_Absent()
    {
        string dir = MakeFolder("model.onnx", "model.uint8.onnx"); // no int8
        try { Assert.Equal("model.uint8.onnx", ModelVariantSelector.SelectVariantFile(dir, "model.onnx", EpKind.Qnn)); }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public void Cuda_Prefers_Fp16()
    {
        string dir = MakeFolder("model.onnx", "model.fp16.onnx");
        try { Assert.Equal("model.fp16.onnx", ModelVariantSelector.SelectVariantFile(dir, "model.onnx", EpKind.Cuda)); }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public void ProviderSupport_Covers_All_NonCpu_Names()
    {
        Assert.True(ModelVariantSelector.IsProviderSupported(new[] { "cuda" }, EpKind.Cuda));
        Assert.True(ModelVariantSelector.IsProviderSupported(new[] { "openvino" }, EpKind.OpenVINO));
        Assert.True(ModelVariantSelector.IsProviderSupported(new[] { "vitisai" }, EpKind.VitisAI));
        Assert.False(ModelVariantSelector.IsProviderSupported(new[] { "cuda" }, EpKind.VitisAI));
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

    [Fact]
    public void Npu_Floor_Is_Exactly_26100()
    {
        // The NPU runtime gate (Task 3.1) hinges on the 24H2 boundary; pin it exactly.
        Assert.False(OsCapabilities.MeetsBuild(26099, OsCapabilities.Build24H2)); // one below → DML/CPU only
        Assert.True(OsCapabilities.MeetsBuild(26100, OsCapabilities.Build24H2));  // floor
        Assert.True(OsCapabilities.MeetsBuild(26101, OsCapabilities.Build24H2));  // above
    }

    [Fact]
    public void Documented_Build_Floors_Are_Stable()
    {
        Assert.Equal(26100, OsCapabilities.Build24H2);
        Assert.Equal(17763, OsCapabilities.Build1809);
    }
}

public class SessionOptionsBuilderTests
{
    [Fact]
    public void IntraOpThreads_Override_Is_Applied()
    {
        using var opts = SessionOptionsBuilder.Build(new EpResolution(EpDeviceInfo.Cpu, false), intraOpThreads: 4);
        Assert.Equal(4, opts.IntraOpNumThreads);
    }

    [Fact]
    public void CompileCachePath_Wiring_Does_Not_Throw()
    {
        // EPContext config entries are set defensively (try/caught across ORT versions); the headless
        // guarantee is "usable options, no throw".
        string ctx = Path.Combine(Path.GetTempPath(), $"ctx_{Guid.NewGuid():N}_ctx.onnx");
        using var opts = SessionOptionsBuilder.Build(new EpResolution(EpDeviceInfo.Cpu, false), 0, ctx);
        Assert.NotNull(opts);
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

public class AutoEpTests
{
    [Fact]
    public void Enumerator_Always_Includes_Cpu()
    {
        var devices = OrtEpEnumerator.Enumerate();
        Assert.Contains(devices, d => d.Kind == EpKind.Cpu);
    }

    [Fact]
    public void DirectML_Requested_Resolves_Dml_Even_Without_Enumerated_Device()
    {
        // DirectML is built into Windows ML, so a DML preference resolves to DML (the builder appends
        // it directly); on a non-GPU host the append is swallowed and CPU executes — but no throw.
        var sel = new ExecutionProviderSelector(enumerateDevices: () => new[] { EpDeviceInfo.Cpu });
        using var opts = sel.BuildSessionOptions(new EpPreference(EpKind.DirectML), "m-1");
        Assert.NotNull(opts);
        Assert.False(sel.LastResolution!.FellBackToCpu);
        Assert.Equal(EpKind.DirectML, sel.LastResolution.Device.Kind);
    }

    [Fact]
    public void NonCpu_Device_Without_Native_Does_Not_Throw()
    {
        // A discovered DirectML device with no backing OrtEpDevice (Native=null) must build clean opts.
        var dml = new EpDeviceInfo("DmlExecutionProvider", HardwareKind.Gpu, EpKind.DirectML);
        var sel = new ExecutionProviderSelector(enumerateDevices: () => new[] { EpDeviceInfo.Cpu, dml });
        using var opts = sel.BuildSessionOptions(new EpPreference(EpKind.DirectML), "m-2");
        Assert.NotNull(opts);
        Assert.False(sel.LastResolution!.FellBackToCpu);
        Assert.Equal(EpKind.DirectML, sel.LastResolution.Device.Kind);
    }

    [Fact]
    public void InvalidateCompiledModel_Deletes_Stale_Ctx_And_Is_Cpu_Safe()
    {
        string dir = Path.Combine(Path.GetTempPath(), $"epcache_{Guid.NewGuid():N}");
        var cache = new CompiledModelCache(dir);
        var dml = new EpDeviceInfo("DmlExecutionProvider", HardwareKind.Gpu, EpKind.DirectML, "1.0", "drv");
        var sel = new ExecutionProviderSelector(enumerateDevices: () => new[] { EpDeviceInfo.Cpu, dml }, cache: cache);

        using (sel.BuildSessionOptions(new EpPreference(EpKind.DirectML), "h")) { }
        string ctx = cache.ContextPath("h", "DmlExecutionProvider", "1.0", "drv");
        File.WriteAllText(ctx, "stale");
        sel.InvalidateCompiledModel("h");
        Assert.False(File.Exists(ctx));

        // CPU resolution has no compiled artifact → invalidation is a harmless no-op.
        using (sel.BuildSessionOptions(new EpPreference(EpKind.Cpu), "h")) { }
        sel.InvalidateCompiledModel("h");
        Directory.Delete(dir, true);
    }
}
