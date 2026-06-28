using Microsoft.ML.OnnxRuntime;
using Stt.Abstractions.Decoders;
using Stt.Abstractions.Ep;
using Stt.Abstractions.Models;
using Stt.Core.Audio;
using Stt.Core.Ep;
using Stt.Core.Features;
using Stt.Core.Pipeline;

namespace Stt.Core.Tests.Decoders;

/// <summary>
/// Real end-to-end offline recognition: a bundled speech WAV → kaldi-fbank (native shim) →
/// SenseVoice (ORT) → text. Skips unless the shim + a SenseVoice model folder + Silero VAD are
/// present (env <c>STT_SENSEVOICE_DIR</c>, <c>STT_SILERO_VAD</c>).
/// </summary>
public class SenseVoiceTranscriptionTests
{
    [SkippableTheory]
    [InlineData("zh.wav", "点")]
    [InlineData("en.wav", "tribal")]
    public void Transcribes_Bundled_Wav(string wavName, string expectedSubstring)
    {
        string? dir = Environment.GetEnvironmentVariable("STT_SENSEVOICE_DIR");
        string? vad = Environment.GetEnvironmentVariable("STT_SILERO_VAD");
        Skip.If(string.IsNullOrEmpty(dir) || !Directory.Exists(dir), "Set STT_SENSEVOICE_DIR to the model folder.");
        Skip.If(string.IsNullOrEmpty(vad) || !File.Exists(vad), "Set STT_SILERO_VAD to silero_vad.onnx.");
        Skip.IfNot(KaldiNativeFbankInterop.IsAvailable, "kaldi_native_fbank_shim not present.");

        var manifest = new ModelManifest
        {
            Id = "sense-voice-small",
            Family = "sense_voice",
            DecoderType = "nar",
            Files = new ModelFiles { Model = "model.int8.onnx", Tokens = "tokens.txt" },
            Feature = new FeatureSpec { Family = "KaldiFbankLfrCmvn", FeatureDim = 560, Lfr = new[] { 7, 6 }, Cmvn = "metadata" },
            Capabilities = new CapabilityFlags { OfflineCapable = true, Multilingual = true, NeedsLfrCmvn = true, NeedsVad = true },
            FolderPath = dir,
        };

        var epSelector = new ExecutionProviderSelector(cache: new CompiledModelCache(Path.GetTempPath()));
        using var chain = OfflinePipelineBuilder.BuildOffline(manifest, vad!, epSelector, new EpPreference(EpKind.Cpu));

        string wavPath = Path.Combine(dir!, "test_wavs", wavName);
        var wav = WavIo.ReadPcm(wavPath);
        float[] mono = Resampler.ToMono16k(wav.Interleaved, wav.SampleRate, wav.Channels);

        float[] feats = chain.Frontend.Extract(mono, out int frames);
        chain.Decoder.AcceptFeatures(feats, frames, chain.Frontend.FeatureDim);
        chain.Decoder.InputFinished();
        string text = chain.Decoder.GetResult().Text;

        File.AppendAllText(Path.Combine(Path.GetTempPath(), "stt_transcript.txt"), $"{wavName} => [{text}]\n");
        Assert.False(string.IsNullOrWhiteSpace(text), $"empty transcript for {wavName}");
        Assert.Contains(expectedSubstring, text);
    }
}
