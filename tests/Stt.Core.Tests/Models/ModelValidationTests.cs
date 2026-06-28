using Stt.Abstractions.Models;
using Stt.Core.Models;
using Stt.Core.Text;

namespace Stt.Core.Tests.Models;

public class ModelValidationTests
{
    private static ModelManifest FbankManifest(int dim = 80, string family = "KaldiFbankPovey") => new()
    {
        Id = "m", Family = "zipformer2",
        Feature = new FeatureSpec { Family = family, FeatureDim = dim },
        Capabilities = new CapabilityFlags { StreamingCapable = true },
    };

    private static ModelProbe FbankProbe(int dim = 80, int vocab = 0) => new()
    {
        ModelType = "zipformer2", FeatureDim = dim, Layout = FeatureLayout.TimeLast, VocabSize = vocab,
    };

    [Fact]
    public void Valid_Fbank_Combo_Passes()
    {
        var report = ModelValidation.Validate(FbankManifest(80), FbankProbe(80), null);
        Assert.True(report.Ok, report.ToString());
    }

    [Fact]
    public void Dim_Mismatch_Is_Rejected_With_Checklist()
    {
        // manifest says 80, graph input is 560 → reject.
        var report = ModelValidation.Validate(FbankManifest(80), FbankProbe(560), null);
        Assert.False(report.Ok);
        Assert.Contains(report.Errors, e => e.Contains("Feature dim mismatch"));
        Assert.NotEmpty(report.Checks); // shows what was inspected
    }

    [Fact]
    public void FamilyB_Missing_Cmvn_Is_Rejected()
    {
        var manifest = new ModelManifest
        {
            Id = "sv", Family = "sense_voice",
            Feature = new FeatureSpec { Family = "KaldiFbankLfrCmvn", FeatureDim = 560, Lfr = new[] { 7, 6 }, Cmvn = "none" },
            Capabilities = new CapabilityFlags { OfflineCapable = true, NeedsLfrCmvn = false },
        };
        var probe = new ModelProbe { ModelType = "sense_voice_ctc", FeatureDim = 560, Layout = FeatureLayout.TimeLast };
        var report = ModelValidation.Validate(manifest, probe, null);
        Assert.False(report.Ok);
        Assert.Contains(report.Errors, e => e.Contains("CMVN"));
    }

    [Fact]
    public void Tokens_Count_Mismatch_Is_Rejected()
    {
        var tokens = TokenTable.Parse(new[] { "a 0", "b 1", "c 2" }); // 3 tokens
        var report = ModelValidation.Validate(FbankManifest(80), FbankProbe(80, vocab: 5000), tokens);
        Assert.False(report.Ok);
        Assert.Contains(report.Errors, e => e.Contains("vocab_size"));
    }

    [Fact]
    public void Unknown_Family_Is_Rejected()
    {
        var manifest = FbankManifest(256, family: "Auto");
        var probe = new ModelProbe { ModelType = "mystery", FeatureDim = 256, Layout = FeatureLayout.Unknown };
        var report = ModelValidation.Validate(manifest, probe, null);
        Assert.False(report.Ok);
        Assert.Contains(report.Errors, e => e.Contains("UNKNOWN"));
    }
}

public class ModelRegistryTests
{
    private static string MakeFolder(Action<string> populate)
    {
        string dir = Path.Combine(Path.GetTempPath(), $"mr_{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        populate(dir);
        return dir;
    }

    [Fact]
    public void Infers_Transducer_From_File_Naming()
    {
        string dir = MakeFolder(d =>
        {
            File.WriteAllText(Path.Combine(d, "encoder.onnx"), "");
            File.WriteAllText(Path.Combine(d, "decoder.onnx"), "");
            File.WriteAllText(Path.Combine(d, "joiner.onnx"), "");
            File.WriteAllText(Path.Combine(d, "tokens.txt"), "a 0\n");
        });
        try
        {
            var reg = new ModelRegistry();
            var m = reg.ImportFromFolder(dir);
            Assert.Equal("transducer", m.Family);
            Assert.True(m.Capabilities.StreamingCapable);
            Assert.False(m.Capabilities.OfflineCapable);
            Assert.Equal("encoder.onnx", m.Files.Encoder);
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public void Infers_SenseVoice_From_Single_Model()
    {
        string dir = MakeFolder(d =>
        {
            File.WriteAllText(Path.Combine(d, "sense-voice.onnx"), "");
            File.WriteAllText(Path.Combine(d, "tokens.txt"), "a 0\n");
        });
        try
        {
            var reg = new ModelRegistry();
            var m = reg.ImportFromFolder(dir);
            Assert.Equal("sense_voice", m.Family);
            Assert.True(m.Capabilities.OfflineCapable);
            Assert.Equal(560, m.Feature.FeatureDim);
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public void ResolveCombination_Rejects_Offline_Model_As_First_Pass()
    {
        var reg = new ModelRegistry();
        // Register a streaming transducer and an offline SenseVoice.
        string sdir = MakeFolder(d =>
        {
            File.WriteAllText(Path.Combine(d, "encoder.onnx"), "");
            File.WriteAllText(Path.Combine(d, "decoder.onnx"), "");
            File.WriteAllText(Path.Combine(d, "joiner.onnx"), "");
        });
        string odir = MakeFolder(d => File.WriteAllText(Path.Combine(d, "sense-voice.onnx"), ""));
        try
        {
            var stream = reg.ImportFromFolder(sdir);
            var offline = reg.ImportFromFolder(odir);

            // Legal: streaming first, offline second.
            var ok = reg.ResolveCombination(stream.Id, offline.Id, Stt.Abstractions.Pipeline.PipelineMode.TwoPass);
            Assert.True(ok.RequiresVad);

            // Illegal: offline model as first pass.
            var ex = Assert.Throws<ArgumentException>(() =>
                reg.ResolveCombination(offline.Id, offline.Id, Stt.Abstractions.Pipeline.PipelineMode.TwoPass));
            Assert.Contains("not streaming-capable", ex.Message);
        }
        finally { Directory.Delete(sdir, true); Directory.Delete(odir, true); }
    }
}
