using Microsoft.ML.OnnxRuntime;
using Stt.Core.Audio;
using Stt.Core.Decoders;
using Stt.Core.Features;
using Stt.Core.Models;
using Stt.Core.Text;

namespace Stt.Core.Tests.Decoders;

/// <summary>
/// Integration test for the streaming Zipformer transducer (spec §8.2 "align to sherpa before
/// trusting"). Skips unless STT_ZIPFORMER_DIR (a folder with encoder.onnx/decoder.onnx/joiner.onnx/
/// tokens.txt) and STT_ZIPFORMER_WAV are set, and the native fbank shim is present. When a sherpa
/// reference transcript is provided via STT_ZIPFORMER_REF, asserts an exact match; otherwise
/// asserts a non-empty hypothesis.
/// </summary>
public class StreamingTransducerTests
{
    [SkippableFact]
    public void Streams_And_Produces_Hypothesis()
    {
        string? dir = Environment.GetEnvironmentVariable("STT_ZIPFORMER_DIR");
        string? wav = Environment.GetEnvironmentVariable("STT_ZIPFORMER_WAV");
        Skip.If(string.IsNullOrEmpty(dir) || string.IsNullOrEmpty(wav), "Set STT_ZIPFORMER_DIR and STT_ZIPFORMER_WAV.");
        Skip.IfNot(KaldiNativeFbankInterop.IsAvailable, "native fbank shim not present.");

        string Enc = Path.Combine(dir!, "encoder.onnx");
        string Dec = Path.Combine(dir!, "decoder.onnx");
        string Joi = Path.Combine(dir!, "joiner.onnx");
        string Tok = Path.Combine(dir!, "tokens.txt");
        Skip.IfNot(File.Exists(Enc) && File.Exists(Dec) && File.Exists(Joi) && File.Exists(Tok), "model files missing.");

        using var encoder = new InferenceSession(Enc);
        using var decoder = new InferenceSession(Dec);
        using var joiner = new InferenceSession(Joi);

        var geo = ZipformerGeometry.FromMetadata(ModelMetadataReader.FromSession(encoder).Metadata);
        var session = new OrtStreamingSession(encoder, decoder, joiner, geo);
        var tokens = TokenTable.Load(Tok);
        using var dec = new TransducerDecoder(session, tokens);

        var audio = WavIo.ReadPcm(wav!);
        float[] mono = Resampler.ToMono16k(audio.Interleaved, audio.SampleRate, audio.Channels);
        using var fe = new KaldiFbankFrontend(FbankOptions.FamilyA(80));
        float[] feats = fe.Extract(mono, out int frames);

        dec.AcceptFeatures(feats, frames, 80);
        dec.InputFinished();
        string hyp = dec.GetResult().Text;

        string? reference = Environment.GetEnvironmentVariable("STT_ZIPFORMER_REF");
        if (!string.IsNullOrEmpty(reference))
            Assert.Equal(reference.Trim(), hyp.Trim());
        else
            Assert.False(string.IsNullOrWhiteSpace(hyp), "expected a non-empty hypothesis.");
    }
}
