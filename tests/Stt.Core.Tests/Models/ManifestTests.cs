using System.Text.Json;
using Stt.Abstractions.Models;

namespace Stt.Core.Tests.Models;

public class ManifestTests
{
    private const string Example = """
    {
      "id": "zipformer-zh-en-streaming",
      "displayName": "Zipformer 中英(流式)",
      "version": "1.0.0",
      "family": "transducer",
      "runtime": ["streaming"],
      "decoderType": "transducer",
      "files": { "encoder":"encoder.onnx", "decoder":"decoder.onnx", "joiner":"joiner.onnx", "tokens":"tokens.txt" },
      "feature": { "frontEnd":"kaldi_fbank", "family":"KaldiFbankPovey", "sampleRate":16000, "featureDim":80, "lfr":null, "cmvn":"none" },
      "capabilities": { "streamingCapable":true, "offlineCapable":false, "needsLfrCmvn":false, "multilingual":true, "emitsTimestamps":true, "needsVad":false },
      "languages": ["zh","en"],
      "decoding": { "defaultMethod":"greedy_search", "endpointRules": { "rule2MinTrailingSilence":1.2 } },
      "providerSupport": ["cpu","directml"],
      "license": "Apache-2.0"
    }
    """;

    [Fact]
    public void Deserializes_Spec_Example()
    {
        var m = JsonSerializer.Deserialize<ModelManifest>(Example)!;
        Assert.Equal("zipformer-zh-en-streaming", m.Id);
        Assert.Equal("transducer", m.Family);
        Assert.Equal(80, m.Feature.FeatureDim);
        Assert.True(m.Capabilities.StreamingCapable);
        Assert.False(m.Capabilities.OfflineCapable);
        Assert.Contains("zh", m.Languages);
        Assert.Equal(1.2, m.Decoding.EndpointRules["rule2MinTrailingSilence"], 3);
        Assert.Equal("encoder.onnx", m.Files.Encoder);
    }

    [Fact]
    public void Round_Trips()
    {
        var m = JsonSerializer.Deserialize<ModelManifest>(Example)!;
        string json = JsonSerializer.Serialize(m);
        var again = JsonSerializer.Deserialize<ModelManifest>(json)!;
        Assert.Equal(m.Id, again.Id);
        Assert.Equal(m.Feature.FeatureDim, again.Feature.FeatureDim);
        Assert.Equal(m.ProviderSupport, again.ProviderSupport);
    }
}
