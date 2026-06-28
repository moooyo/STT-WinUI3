namespace Stt.Core.Ep;

/// <summary>
/// Manages EPContext compiled-model cache files (spec §9, §14). The compiled graph is written to
/// the local cache folder with a filename stamped by model hash + EP name + EP version + driver,
/// so a driver/EP update produces a different path and the stale artifact is ignored rather than
/// loaded (which would fail with INVALID_GRAPH). Treating "recompile" as normal, not fatal.
/// </summary>
public sealed class CompiledModelCache
{
    private readonly string _root;

    public CompiledModelCache(string cacheRoot)
    {
        _root = cacheRoot;
        Directory.CreateDirectory(_root);
    }

    public string CacheRoot => _root;

    /// <summary>Deterministic stamped path: <c>{hash}_{ep}_{ver}_{driver}_ctx.onnx</c>.</summary>
    public string ContextPath(string modelHash, string epName, string epVersion, string driver)
    {
        string name = $"{Sanitize(modelHash)}_{Sanitize(epName)}_{Sanitize(epVersion)}_{Sanitize(driver)}_ctx.onnx";
        return Path.Combine(_root, name);
    }

    /// <summary>True if a non-empty compiled artifact exists at the stamped path.</summary>
    public bool TryGetValid(string modelHash, string epName, string epVersion, string driver, out string path)
    {
        path = ContextPath(modelHash, epName, epVersion, driver);
        return File.Exists(path) && new FileInfo(path).Length > 0;
    }

    /// <summary>Delete a stale compiled artifact (idempotent).</summary>
    public void Invalidate(string path)
    {
        try { if (File.Exists(path)) File.Delete(path); }
        catch (IOException) { /* best-effort; recompile will overwrite */ }
    }

    private static string Sanitize(string s)
    {
        if (string.IsNullOrEmpty(s)) return "none";
        Span<char> buf = stackalloc char[s.Length];
        for (int i = 0; i < s.Length; i++)
        {
            char c = s[i];
            buf[i] = (char.IsLetterOrDigit(c) || c is '-' or '.') ? c : '-';
        }
        return new string(buf);
    }
}
