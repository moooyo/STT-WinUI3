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
        var devices = new[] { EpDeviceInfo.Cpu }; // no DML present
        var res = EpResolver.Resolve(new EpPreference(EpKind.DirectML, AllowFallbackToCpu: true), devices);
        Assert.Equal(EpKind.Cpu, res.Device.Kind);
        Assert.True(res.FellBackToCpu);
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
    public void Falls_Back_To_Cpu_When_Dml_Enumerated_But_Absent()
    {
        var selector = new ExecutionProviderSelector(enumerateDevices: () => new[] { EpDeviceInfo.Cpu });
        using var opts = selector.BuildSessionOptions(new EpPreference(EpKind.DirectML), "hash2");
        Assert.True(selector.LastResolution!.FellBackToCpu);
    }
}
