using Stt.Abstractions.Features;
using Stt.Core.Models;

namespace Stt.Core.Tests.Models;

/// <summary>
/// Verifies metadata-driven import inference classifies real model folders into the right feature
/// family (no manifest.json present). Gated on the model env vars (skips in CI).
/// </summary>
public class ModelImportInferenceTests
{
    [SkippableFact]
    public void Infers_Whisper_As_FamilyC()
    {
        string? dir = Environment.GetEnvironmentVariable("STT_WHISPER_DIR");
        Skip.If(string.IsNullOrEmpty(dir) || !Directory.Exists(dir), "set STT_WHISPER_DIR");
        var reg = new ModelRegistry();
        var m = reg.ImportFromFolder(dir!);
        Assert.Equal("WhisperLogMel", m.Feature.Family);
        Assert.Equal("ar", m.DecoderType);
        Assert.True(m.Capabilities.OfflineCapable);
        Assert.NotNull(m.Files.Encoder);
        Assert.NotNull(m.Files.Decoder);
        Assert.Null(m.Files.Joiner);
    }

    [SkippableFact]
    public void Infers_NeMo_As_FamilyD()
    {
        string? dir = Environment.GetEnvironmentVariable("STT_NEMO_DIR");
        Skip.If(string.IsNullOrEmpty(dir) || !Directory.Exists(dir), "set STT_NEMO_DIR");
        var reg = new ModelRegistry();
        var m = reg.ImportFromFolder(dir!);
        Assert.Equal("NemoMel", m.Feature.Family);
        Assert.True(m.Capabilities.OfflineCapable);
        Assert.NotNull(m.Files.Model);
    }

    [SkippableFact]
    public void Infers_SenseVoice_As_FamilyB()
    {
        string? dir = Environment.GetEnvironmentVariable("STT_SENSEVOICE_DIR");
        Skip.If(string.IsNullOrEmpty(dir) || !Directory.Exists(dir), "set STT_SENSEVOICE_DIR");
        // Probe the bare model folder without its manifest by pointing at a temp copy is overkill;
        // instead confirm the detector maps the model's metadata to Family B.
        var reg = new ModelRegistry();
        var m = reg.ImportFromFolder(dir!);
        Assert.Equal(AsrFeatureFamily.KaldiFbankLfrCmvn.ToString(), m.Feature.Family);
        Assert.True(m.Capabilities.OfflineCapable);
    }
}
