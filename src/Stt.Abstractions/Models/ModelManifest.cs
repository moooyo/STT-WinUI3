using System.Text.Json.Serialization;

namespace Stt.Abstractions.Models;

/// <summary>Paths (relative to the model folder) of a model's ONNX graphs + sidecar files (spec §5.3).</summary>
public sealed record ModelFiles
{
    [JsonPropertyName("encoder")] public string? Encoder { get; init; }
    [JsonPropertyName("decoder")] public string? Decoder { get; init; }
    [JsonPropertyName("joiner")] public string? Joiner { get; init; }
    [JsonPropertyName("model")] public string? Model { get; init; }     // single-graph CTC/NAR
    [JsonPropertyName("tokens")] public string? Tokens { get; init; }
}

/// <summary>Feature front-end spec (spec §5.3 "feature").</summary>
public sealed record FeatureSpec
{
    /// <summary>e.g. "kaldi_fbank", "whisper_mel".</summary>
    [JsonPropertyName("frontEnd")] public string FrontEnd { get; init; } = "kaldi_fbank";

    /// <summary>Feature family name, mapped to <c>AsrFeatureFamily</c>.</summary>
    [JsonPropertyName("family")] public string Family { get; init; } = "KaldiFbankPovey";

    [JsonPropertyName("sampleRate")] public int SampleRate { get; init; } = 16000;
    [JsonPropertyName("featureDim")] public int FeatureDim { get; init; } = 80;

    /// <summary>LFR window/shift as <c>[m, n]</c>, or null when not used (Family B uses [7,6]).</summary>
    [JsonPropertyName("lfr")] public int[]? Lfr { get; init; }

    /// <summary>CMVN mode: "none" or a stats file reference; constants usually read from metadata.</summary>
    [JsonPropertyName("cmvn")] public string Cmvn { get; init; } = "none";
}

/// <summary>Capability flags (spec §5.3 "capabilities"). Drive UI legality + pipeline validation.</summary>
public sealed record CapabilityFlags
{
    [JsonPropertyName("streamingCapable")] public bool StreamingCapable { get; init; }
    [JsonPropertyName("offlineCapable")] public bool OfflineCapable { get; init; }
    [JsonPropertyName("needsLfrCmvn")] public bool NeedsLfrCmvn { get; init; }
    [JsonPropertyName("multilingual")] public bool Multilingual { get; init; }
    [JsonPropertyName("emitsTimestamps")] public bool EmitsTimestamps { get; init; }
    [JsonPropertyName("needsVad")] public bool NeedsVad { get; init; }
}

/// <summary>Decoding options (spec §5.3 "decoding").</summary>
public sealed record DecodingSpec
{
    [JsonPropertyName("defaultMethod")] public string DefaultMethod { get; init; } = "greedy_search";

    /// <summary>Endpoint rules; <c>rule2MinTrailingSilence</c> is the primary one (seconds).</summary>
    [JsonPropertyName("endpointRules")] public Dictionary<string, double> EndpointRules { get; init; } = new();
}

/// <summary>
/// A model descriptor (spec §5.3). Most fields auto-derive from ONNX <c>metadata_props</c>
/// (spec §10); missing ones are supplied by the user/sideload manifest.
/// </summary>
public sealed record ModelManifest
{
    [JsonPropertyName("id")] public string Id { get; init; } = "";
    [JsonPropertyName("displayName")] public string DisplayName { get; init; } = "";
    [JsonPropertyName("version")] public string Version { get; init; } = "1.0.0";

    /// <summary>Architecture family: transducer|paraformer|zipformer2_ctc|whisper|sense_voice|...</summary>
    [JsonPropertyName("family")] public string Family { get; init; } = "";

    /// <summary>Runtime modes: "streaming" | "offline" | both.</summary>
    [JsonPropertyName("runtime")] public string[] Runtime { get; init; } = Array.Empty<string>();

    /// <summary>transducer|ctc|nar|ar.</summary>
    [JsonPropertyName("decoderType")] public string DecoderType { get; init; } = "";

    [JsonPropertyName("files")] public ModelFiles Files { get; init; } = new();
    [JsonPropertyName("feature")] public FeatureSpec Feature { get; init; } = new();
    [JsonPropertyName("capabilities")] public CapabilityFlags Capabilities { get; init; } = new();
    [JsonPropertyName("languages")] public string[] Languages { get; init; } = Array.Empty<string>();
    [JsonPropertyName("decoding")] public DecodingSpec Decoding { get; init; } = new();

    /// <summary>EP names this model is valid on; models with dynamic shapes omit "qnn".</summary>
    [JsonPropertyName("providerSupport")] public string[] ProviderSupport { get; init; } = Array.Empty<string>();

    [JsonPropertyName("license")] public string License { get; init; } = "";

    /// <summary>Absolute folder this manifest was loaded from (set at import; not serialized).</summary>
    [JsonIgnore] public string? FolderPath { get; init; }
}
