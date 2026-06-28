using System.Text.Json;
using Stt.Abstractions.Models;

namespace Stt.Core.Models;

/// <summary>Load/save <see cref="ModelManifest"/> as camelCase JSON (spec §5.3).</summary>
public static class ModelManifestIo
{
    private static readonly JsonSerializerOptions Options = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        WriteIndented = true,
        ReadCommentHandling = JsonCommentHandling.Skip,
        AllowTrailingCommas = true,
    };

    public static ModelManifest Load(string path)
    {
        string json = File.ReadAllText(path);
        return JsonSerializer.Deserialize<ModelManifest>(json, Options)
               ?? throw new InvalidDataException($"Empty or invalid manifest: {path}");
    }

    public static void Save(ModelManifest manifest, string path)
    {
        File.WriteAllText(path, JsonSerializer.Serialize(manifest, Options));
    }
}
