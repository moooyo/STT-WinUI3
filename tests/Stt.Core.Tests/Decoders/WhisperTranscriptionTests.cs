using Stt.Abstractions.Ep;
using Stt.Abstractions.Models;
using Stt.Core.Audio;
using Stt.Core.Ep;
using Stt.Core.Features;
using Stt.Core.Pipeline;

namespace Stt.Core.Tests.Decoders;

/// <summary>
/// Real end-to-end test of Family C (Whisper) through the production pipeline builder: a bundled
/// speech WAV → OfflinePipelineBuilder (WhisperMelFrontend + WhisperArDecoder: encoder + greedy AR
/// decoder with KV cache) → text. Proves Whisper is first-class via the same path the app uses.
/// Skips unless the shim + a sherpa-onnx Whisper model folder + a Silero VAD are present
/// (env STT_WHISPER_DIR, STT_SILERO_VAD).
/// </summary>
public class WhisperTranscriptionTests
{
    [SkippableTheory]
    [InlineData("0.wav", "nightfall")]
    [InlineData("1.wav", "consequence")]
    public void Transcribes_Whisper_Via_Pipeline(string wavName, string expectedSubstring)
    {
        string? dir = Environment.GetEnvironmentVariable("STT_WHISPER_DIR");
        string? vad = Environment.GetEnvironmentVariable("STT_SILERO_VAD");
        Skip.If(string.IsNullOrEmpty(dir) || !Directory.Exists(dir), "Set STT_WHISPER_DIR to a sherpa-onnx Whisper model folder.");
        Skip.If(string.IsNullOrEmpty(vad) || !File.Exists(vad), "Set STT_SILERO_VAD to silero_vad.onnx.");
        Skip.IfNot(KaldiNativeFbankInterop.IsAvailable, "kaldi_native_fbank_shim not present.");

        string encName = Path.GetFileName(Directory.GetFiles(dir!, "*-encoder.onnx").FirstOrDefault()
            ?? throw new FileNotFoundException("no *-encoder.onnx in STT_WHISPER_DIR"));
        string prefix = encName[..^"-encoder.onnx".Length];

        var manifest = new ModelManifest
        {
            Id = "whisper",
            Family = "whisper",
            DecoderType = "ar",
            Files = new ModelFiles
            {
                Encoder = $"{prefix}-encoder.onnx",
                Decoder = $"{prefix}-decoder.onnx",
                Tokens = $"{prefix}-tokens.txt",
            },
            Feature = new FeatureSpec { Family = "WhisperLogMel", FeatureDim = 80 },
            Capabilities = new CapabilityFlags { OfflineCapable = true, Multilingual = true, NeedsVad = true },
            FolderPath = dir,
        };

        var epSelector = new ExecutionProviderSelector(cache: new CompiledModelCache(Path.GetTempPath()));
        using var chain = OfflinePipelineBuilder.BuildOffline(manifest, vad!, epSelector, new EpPreference(EpKind.Cpu));

        Assert.IsType<WhisperMelFrontend>(chain.Frontend);

        var wav = WavIo.ReadPcm(Path.Combine(dir!, "test_wavs", wavName));
        float[] mono = Resampler.ToMono16k(wav.Interleaved, wav.SampleRate, wav.Channels);

        float[] feats = chain.Frontend.Extract(mono, out int frames);
        chain.Decoder.AcceptFeatures(feats, frames, chain.Frontend.FeatureDim);
        chain.Decoder.InputFinished();
        string text = chain.Decoder.GetResult().Text;

        File.AppendAllText(Path.Combine(Path.GetTempPath(), "stt_whisper.txt"), $"{wavName} => [{text}]\n");
        Assert.Contains(expectedSubstring, text, StringComparison.OrdinalIgnoreCase);
    }
}
