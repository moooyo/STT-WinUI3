using Stt.Abstractions.Ep;
using Stt.Abstractions.Models;
using Stt.Core.Audio;
using Stt.Core.Ep;
using Stt.Core.Features;
using Stt.Core.Pipeline;

namespace Stt.Core.Tests.Decoders;

/// <summary>
/// Real end-to-end test of Family D (NeMo) through the production pipeline builder: a bundled speech
/// WAV → OfflinePipelineBuilder (NemoMelFrontend + OrtNarModelRunner with mel-major layout + NarDecoder
/// with the detected CTC blank) → text, asserting the expected transcript. Proves a NeMo Conformer-CTC
/// model is first-class via the same path the app uses. Skips unless the shim + a NeMo CTC model folder
/// + a Silero VAD are present (env STT_NEMO_DIR, STT_SILERO_VAD).
/// </summary>
public class NemoCtcTranscriptionTests
{
    [SkippableTheory]
    [InlineData("0.wav", "nightfall")]
    [InlineData("1.wav", "consequence")]
    public void Transcribes_NeMo_Ctc_Via_Pipeline(string wavName, string expectedSubstring)
    {
        string? dir = Environment.GetEnvironmentVariable("STT_NEMO_DIR");
        string? vad = Environment.GetEnvironmentVariable("STT_SILERO_VAD");
        Skip.If(string.IsNullOrEmpty(dir) || !Directory.Exists(dir), "Set STT_NEMO_DIR to a NeMo CTC model folder.");
        Skip.If(string.IsNullOrEmpty(vad) || !File.Exists(vad), "Set STT_SILERO_VAD to silero_vad.onnx.");
        Skip.IfNot(KaldiNativeFbankInterop.IsAvailable, "kaldi_native_fbank_shim not present.");

        var manifest = new ModelManifest
        {
            Id = "nemo-conformer-ctc",
            Family = "nemo",
            DecoderType = "ctc",
            Files = new ModelFiles { Model = "model.onnx", Tokens = "tokens.txt" },
            Feature = new FeatureSpec { Family = "NemoMel", FeatureDim = 80 },
            Capabilities = new CapabilityFlags { OfflineCapable = true, Multilingual = false, NeedsVad = true },
            FolderPath = dir,
        };

        var epSelector = new ExecutionProviderSelector(cache: new CompiledModelCache(Path.GetTempPath()));
        using var chain = OfflinePipelineBuilder.BuildOffline(manifest, vad!, epSelector, new EpPreference(EpKind.Cpu));

        Assert.IsType<NemoMelFrontend>(chain.Frontend);

        var wav = WavIo.ReadPcm(Path.Combine(dir!, "test_wavs", wavName));
        float[] mono = Resampler.ToMono16k(wav.Interleaved, wav.SampleRate, wav.Channels);

        float[] feats = chain.Frontend.Extract(mono, out int frames);
        chain.Decoder.AcceptFeatures(feats, frames, chain.Frontend.FeatureDim);
        chain.Decoder.InputFinished();
        string text = chain.Decoder.GetResult().Text;

        File.AppendAllText(Path.Combine(Path.GetTempPath(), "stt_nemo.txt"), $"{wavName} => [{text}]\n");
        Assert.Contains(expectedSubstring, text, StringComparison.OrdinalIgnoreCase);
    }
}
