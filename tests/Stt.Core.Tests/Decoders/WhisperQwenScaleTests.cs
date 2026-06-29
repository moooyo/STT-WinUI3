using Microsoft.ML.OnnxRuntime;
using Stt.Abstractions.Ep;
using Stt.Abstractions.Features;
using Stt.Abstractions.Models;
using Stt.Core.Audio;
using Stt.Core.Ep;
using Stt.Core.Features;
using Stt.Core.Models;
using Stt.Core.Pipeline;

namespace Stt.Core.Tests.Decoders;

/// <summary>
/// Scale test for Family C beyond tiny.en: a large multilingual Whisper (base/small/large) and a
/// Qwen-ASR scaffold. The n_mels bin count is taken from encoder metadata (80 ≤ large-v2, 128
/// large-v3, model-defined for Qwen) instead of hardcoding 80, proving the front-end isn't tied to a
/// single variant. Skips unless the model + VAD env vars are set, so headless CI stays green.
/// </summary>
public class WhisperQwenScaleTests
{
    private static int DetectNMels(string encPath)
    {
        using var enc = new InferenceSession(encPath);
        var probe = ModelMetadataReader.FromSession(enc);
        return probe.NMels ?? (probe.FeatureDim > 0 ? probe.FeatureDim : 80);
    }

    [SkippableFact]
    public void Multilingual_Whisper_Transcribes_NonEnglish()
    {
        string? dir = Environment.GetEnvironmentVariable("STT_WHISPER_ML_DIR");
        string? wav = Environment.GetEnvironmentVariable("STT_WHISPER_ML_WAV");
        string? vad = Environment.GetEnvironmentVariable("STT_SILERO_VAD");
        Skip.If(string.IsNullOrEmpty(dir) || string.IsNullOrEmpty(wav) || string.IsNullOrEmpty(vad),
            "Set STT_WHISPER_ML_DIR (multilingual whisper folder), STT_WHISPER_ML_WAV, STT_SILERO_VAD.");
        Skip.IfNot(KaldiNativeFbankInterop.IsAvailable, "kaldi_native_fbank_shim not present.");

        string enc = Path.GetFileName(Directory.GetFiles(dir!, "*-encoder.onnx").First());
        string prefix = enc[..^"-encoder.onnx".Length];
        int nMels = DetectNMels(Path.Combine(dir!, enc));   // 80 or 128 — metadata-driven, not assumed

        var manifest = new ModelManifest
        {
            Id = "whisper-ml", Family = "whisper", DecoderType = "ar",
            Files = new ModelFiles { Encoder = enc, Decoder = $"{prefix}-decoder.onnx", Tokens = $"{prefix}-tokens.txt" },
            Feature = new FeatureSpec { Family = "WhisperLogMel", FeatureDim = nMels },
            Capabilities = new CapabilityFlags { OfflineCapable = true, Multilingual = true, NeedsVad = true },
            FolderPath = dir,
        };
        var sel = new ExecutionProviderSelector(cache: new CompiledModelCache(Path.GetTempPath()));
        using var chain = OfflinePipelineBuilder.BuildOffline(manifest, vad!, sel, new EpPreference(EpKind.Cpu));

        var w = WavIo.ReadPcm(wav!);
        float[] mono = Resampler.ToMono16k(w.Interleaved, w.SampleRate, w.Channels);
        float[] feats = chain.Frontend.Extract(mono, out int frames);
        chain.Decoder.AcceptFeatures(feats, frames, chain.Frontend.FeatureDim);
        chain.Decoder.InputFinished();
        string text = chain.Decoder.GetResult().Text;

        Assert.False(string.IsNullOrWhiteSpace(text), "multilingual whisper produced empty text");
        string? reference = Environment.GetEnvironmentVariable("STT_WHISPER_ML_REF");
        if (!string.IsNullOrEmpty(reference)) Assert.Contains(reference.Trim(), text, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Qwen_Model_Type_Routes_To_WhisperLogMel()
    {
        // Scaffold: a qwen-typed log-mel export classifies as Family C (shared front-end + AR decoder).
        var probe = ModelMetadataReader.Read(
            new Dictionary<string, string> { ["model_type"] = "qwen-asr", ["n_mels"] = "128" },
            new Dictionary<string, int[]> { ["input_features"] = new[] { 1, 128, 3000 } });
        Assert.Equal(AsrFeatureFamily.WhisperLogMel, FeatureFamilyDetector.Detect(probe));
    }

    [SkippableFact]
    public void Qwen_Asr_Transcribes_If_Onnx_Present()
    {
        string? dir = Environment.GetEnvironmentVariable("STT_QWEN_DIR");
        string? wav = Environment.GetEnvironmentVariable("STT_QWEN_WAV");
        string? vad = Environment.GetEnvironmentVariable("STT_SILERO_VAD");
        Skip.If(string.IsNullOrEmpty(dir) || string.IsNullOrEmpty(wav) || string.IsNullOrEmpty(vad),
            "Set STT_QWEN_DIR (encoder/decoder/tokens), STT_QWEN_WAV, STT_SILERO_VAD — Qwen ASR ONNX is rare.");
        Skip.IfNot(KaldiNativeFbankInterop.IsAvailable, "kaldi_native_fbank_shim not present.");

        string enc = Path.GetFileName(Directory.GetFiles(dir!, "*encoder*.onnx").First());
        string prefix = enc[..enc.IndexOf("encoder", StringComparison.OrdinalIgnoreCase)];
        int nMels = DetectNMels(Path.Combine(dir!, enc));
        var manifest = new ModelManifest
        {
            Id = "qwen", Family = "whisper", DecoderType = "ar",
            Files = new ModelFiles { Encoder = enc, Decoder = $"{prefix}decoder.onnx", Tokens = $"{prefix}tokens.txt" },
            Feature = new FeatureSpec { Family = "WhisperLogMel", FeatureDim = nMels },
            Capabilities = new CapabilityFlags { OfflineCapable = true, Multilingual = true, NeedsVad = true },
            FolderPath = dir,
        };
        var sel = new ExecutionProviderSelector(cache: new CompiledModelCache(Path.GetTempPath()));
        using var chain = OfflinePipelineBuilder.BuildOffline(manifest, vad!, sel, new EpPreference(EpKind.Cpu));
        var w = WavIo.ReadPcm(wav!);
        float[] mono = Resampler.ToMono16k(w.Interleaved, w.SampleRate, w.Channels);
        float[] feats = chain.Frontend.Extract(mono, out int frames);
        chain.Decoder.AcceptFeatures(feats, frames, chain.Frontend.FeatureDim);
        chain.Decoder.InputFinished();
        Assert.False(string.IsNullOrWhiteSpace(chain.Decoder.GetResult().Text));
    }
}
