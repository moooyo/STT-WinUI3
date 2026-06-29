using Stt.Abstractions.Ep;
using Stt.Core.Ep;

namespace Stt.Core.Tests.Ep;

public class CompiledModelCacheTests
{
    [Fact]
    public void ContextPath_Is_Deterministic_And_Stamped()
    {
        string root = Path.Combine(Path.GetTempPath(), $"cmc_{Guid.NewGuid():N}");
        try
        {
            var cache = new CompiledModelCache(root);
            string p1 = cache.ContextPath("abc123", "DmlExecutionProvider", "1.20", "31.0.101");
            string p2 = cache.ContextPath("abc123", "DmlExecutionProvider", "1.20", "31.0.101");
            Assert.Equal(p1, p2);
            Assert.Contains("abc123", p1);
            Assert.Contains("DmlExecutionProvider", p1);
            Assert.EndsWith("_ctx.onnx", p1);

            // Different driver → different path (stale artifact ignored, not loaded).
            string p3 = cache.ContextPath("abc123", "DmlExecutionProvider", "1.20", "32.0.000");
            Assert.NotEqual(p1, p3);
        }
        finally { if (Directory.Exists(root)) Directory.Delete(root, true); }
    }

    [Fact]
    public void TryGetValid_False_When_Missing_True_When_Present()
    {
        string root = Path.Combine(Path.GetTempPath(), $"cmc_{Guid.NewGuid():N}");
        try
        {
            var cache = new CompiledModelCache(root);
            Assert.False(cache.TryGetValid("h", "ep", "v", "d", out string path));

            File.WriteAllBytes(path, new byte[] { 1, 2, 3 });
            Assert.True(cache.TryGetValid("h", "ep", "v", "d", out _));

            cache.Invalidate(path);
            Assert.False(cache.TryGetValid("h", "ep", "v", "d", out _));
        }
        finally { if (Directory.Exists(root)) Directory.Delete(root, true); }
    }
}

public class EpResolverTests
{
    private static readonly EpDeviceInfo Dml =
        new("DmlExecutionProvider", HardwareKind.Gpu, EpKind.DirectML, "1.20", "31.0");

    [Fact]
    public void Falls_Back_To_Cpu_When_Requested_Device_Absent()
    {
        var devices = new[] { EpDeviceInfo.Cpu }; // no NPU present
        var res = EpResolver.Resolve(new EpPreference(EpKind.Qnn, AllowFallbackToCpu: true), devices);
        Assert.Equal(EpKind.Cpu, res.Device.Kind);
        Assert.True(res.FellBackToCpu);
    }

    [Fact]
    public void DirectML_Resolves_Even_When_Not_Enumerated()
    {
        // DirectML is built into Windows ML, so it's honored without an enumerated device.
        var res = EpResolver.Resolve(new EpPreference(EpKind.DirectML), new[] { EpDeviceInfo.Cpu });
        Assert.Equal(EpKind.DirectML, res.Device.Kind);
        Assert.False(res.FellBackToCpu);
    }

    [Fact]
    public void Selects_Requested_Device_When_Present()
    {
        var devices = new[] { EpDeviceInfo.Cpu, Dml };
        var res = EpResolver.Resolve(new EpPreference(EpKind.DirectML), devices);
        Assert.Equal(EpKind.DirectML, res.Device.Kind);
        Assert.False(res.FellBackToCpu);
    }

    [Fact]
    public void Throws_When_Absent_And_Fallback_Disabled()
    {
        var devices = new[] { EpDeviceInfo.Cpu };
        Assert.Throws<InvalidOperationException>(() =>
            EpResolver.Resolve(new EpPreference(EpKind.Qnn, AllowFallbackToCpu: false), devices));
    }

    [Fact]
    public void Cpu_Preference_Never_Reports_Fallback()
    {
        var res = EpResolver.Resolve(new EpPreference(EpKind.Cpu), new[] { EpDeviceInfo.Cpu });
        Assert.False(res.FellBackToCpu);
    }

    [Fact]
    public void Auto_Preference_Resolves_To_Cpu_Without_Fallback()
    {
        // EpPreference.Auto == (Cpu, AllowFallbackToCpu:true): a CPU preference is not a "fallback".
        var res = EpResolver.Resolve(EpPreference.Auto, new[] { EpDeviceInfo.Cpu });
        Assert.Equal(EpKind.Cpu, res.Device.Kind);
        Assert.False(res.FellBackToCpu);
    }

    [Theory]
    [InlineData(EpKind.Cuda)]
    [InlineData(EpKind.OpenVINO)]
    [InlineData(EpKind.VitisAI)]
    public void All_NonCpu_Kinds_Fall_Back_When_Absent(EpKind kind)
    {
        var res = EpResolver.Resolve(new EpPreference(kind, AllowFallbackToCpu: true), new[] { EpDeviceInfo.Cpu });
        Assert.Equal(EpKind.Cpu, res.Device.Kind);
        Assert.True(res.FellBackToCpu);
    }

    [Fact]
    public void Selects_Requested_Device_Among_Multiple_NonCpu()
    {
        var npu = new EpDeviceInfo("QNNExecutionProvider", HardwareKind.Npu, EpKind.Qnn, "2.0", "1.0");
        var devices = new[] { EpDeviceInfo.Cpu, Dml, npu };
        var res = EpResolver.Resolve(new EpPreference(EpKind.Qnn), devices);
        Assert.Equal(EpKind.Qnn, res.Device.Kind);
        Assert.False(res.FellBackToCpu);
    }
}

public class ExecutionProviderSelectorTests
{
    [Fact]
    public void Builds_Cpu_Options_By_Default()
    {
        var selector = new ExecutionProviderSelector();
        using var opts = selector.BuildSessionOptions(EpPreference.Auto, "hash1");
        Assert.NotNull(opts);
        Assert.NotNull(selector.LastResolution);
        Assert.Equal(EpKind.Cpu, selector.LastResolution!.Device.Kind);
    }

    [Fact]
    public void Honors_Dml_Even_When_Not_Enumerated()
    {
        // DirectML is built into Windows ML — requesting it resolves to DML (direct append), not CPU.
        var selector = new ExecutionProviderSelector(enumerateDevices: () => new[] { EpDeviceInfo.Cpu });
        using var opts = selector.BuildSessionOptions(new EpPreference(EpKind.DirectML), "hash2");
        Assert.False(selector.LastResolution!.FellBackToCpu);
        Assert.Equal(EpKind.DirectML, selector.LastResolution.Device.Kind);
    }

    private static readonly EpDeviceInfo Dml =
        new("DmlExecutionProvider", HardwareKind.Gpu, EpKind.DirectML, "1.20", "31.0.101");

    [Fact]
    public void Selects_Dml_When_Device_Enumerated_Present()
    {
        var selector = new ExecutionProviderSelector(enumerateDevices: () => new[] { EpDeviceInfo.Cpu, Dml });
        using var opts = selector.BuildSessionOptions(new EpPreference(EpKind.DirectML), "hash3");
        Assert.NotNull(opts);
        Assert.Equal(EpKind.DirectML, selector.LastResolution!.Device.Kind);
        Assert.False(selector.LastResolution!.FellBackToCpu);
    }

    [Fact]
    public void Computes_Context_Cache_Path_For_NonCpu_Device_Only()
    {
        string root = Path.Combine(Path.GetTempPath(), $"epc_{Guid.NewGuid():N}");
        try
        {
            var cache = new CompiledModelCache(root);

            // Present DML device → exercises the `_cache != null && Device.Kind != Cpu` cache-path branch.
            var dmlSel = new ExecutionProviderSelector(enumerateDevices: () => new[] { EpDeviceInfo.Cpu, Dml }, cache: cache);
            using var dmlOpts = dmlSel.BuildSessionOptions(new EpPreference(EpKind.DirectML), "h");
            Assert.NotNull(dmlOpts);
            Assert.Equal(EpKind.DirectML, dmlSel.LastResolution!.Device.Kind);

            // CPU resolution → cache branch is skipped (no throw, CPU options).
            var cpuSel = new ExecutionProviderSelector(enumerateDevices: () => new[] { EpDeviceInfo.Cpu }, cache: cache);
            using var cpuOpts = cpuSel.BuildSessionOptions(EpPreference.Auto, "h");
            Assert.NotNull(cpuOpts);
            Assert.Equal(EpKind.Cpu, cpuSel.LastResolution!.Device.Kind);
        }
        finally { if (Directory.Exists(root)) Directory.Delete(root, true); }
    }
}
