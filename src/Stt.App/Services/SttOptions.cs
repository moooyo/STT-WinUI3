using System.Text.Json;
using Stt.Abstractions.Ep;
using Stt.Abstractions.Pipeline;

namespace Stt.App.Services;

/// <summary>
/// User-configurable engine settings (spec §12 pipeline/settings page), persisted as JSON under
/// LocalAppData. Maps to a <see cref="PipelineConfig"/> when starting transcription.
/// </summary>
public sealed class SttOptions
{
    public PipelineMode Mode { get; set; } = PipelineMode.OnePassOffline; // Phase 0 default
    public string? FirstPassModelId { get; set; }
    public string? SecondPassModelId { get; set; }
    public EpKind Ep { get; set; } = EpKind.Cpu;
    public string? VadModelPath { get; set; }
    public float MinTrailingSilenceSeconds { get; set; } = 0.8f;
    public float MaxUtteranceSeconds { get; set; } = 20f;

    /// <summary>Folders of sideloaded models the user imported — re-scanned at startup so imports persist.</summary>
    public List<string> ImportedModelPaths { get; set; } = new();

    private static string FilePath =>
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Stt", "settings.json");

    public PipelineConfig ToPipelineConfig() => new()
    {
        Mode = Mode,
        FirstPassModelId = FirstPassModelId,
        SecondPassModelId = SecondPassModelId,
        Ep = new EpPreference(Ep),
        MinTrailingSilenceSeconds = MinTrailingSilenceSeconds,
        MaxUtteranceSeconds = MaxUtteranceSeconds,
    };

    public void Save()
    {
        Directory.CreateDirectory(Path.GetDirectoryName(FilePath)!);
        File.WriteAllText(FilePath, JsonSerializer.Serialize(this, new JsonSerializerOptions { WriteIndented = true }));
    }

    public static SttOptions Load()
    {
        try
        {
            if (File.Exists(FilePath))
                return JsonSerializer.Deserialize<SttOptions>(File.ReadAllText(FilePath)) ?? new SttOptions();
        }
        catch { /* corrupt settings → defaults */ }
        return new SttOptions();
    }
}
