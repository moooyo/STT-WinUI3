using Microsoft.ML.OnnxRuntime;
using Stt.Core.Audio;
using Stt.Core.Decoders;
using Stt.Core.Features;

namespace Stt.Core.Tests.Decoders;

/// <summary>
/// Qwen3-ASR (community ONNX) end-to-end: WhisperMelFrontend(128) → encoder → decoder_init/step
/// (Family C, autoregressive LLM). Gated on STT_QWEN_DIR (folder with encoder*.onnx,
/// decoder_init*.onnx, decoder_step*.onnx, embed_tokens.bin, vocab.json) + STT_QWEN_WAV. CPU here;
/// DirectML in the app. Optional STT_QWEN_REF asserts a substring.
/// </summary>
public class QwenAsrTranscriptionTests
{
    [SkippableFact]
    public void Transcribes_Qwen_Via_Three_Graphs()
    {
        string? dir = Environment.GetEnvironmentVariable("STT_QWEN_DIR");
        string? wav = Environment.GetEnvironmentVariable("STT_QWEN_WAV");
        Skip.If(string.IsNullOrEmpty(dir) || string.IsNullOrEmpty(wav), "Set STT_QWEN_DIR + STT_QWEN_WAV.");
        Skip.IfNot(KaldiNativeFbankInterop.IsAvailable, "fbank shim not present.");

        string Pick(string g) => Directory.GetFiles(dir!, g).OrderBy(f => f.Length).First();
        using var enc = new InferenceSession(Pick("encoder*.onnx"));
        using var init = new InferenceSession(Pick("decoder_init*.onnx"));
        using var step = new InferenceSession(Pick("decoder_step*.onnx"));
        float[] embed = QwenAsrDecoder.LoadEmbedTokens(Path.Combine(dir!, "embed_tokens.bin"));
        var detok = QwenAsrDecoder.LoadDetok(Path.Combine(dir!, "vocab.json"));

        using var fe = new WhisperMelFrontend(128);
        using var dec = new QwenAsrDecoder(enc, init, step, embed, 1024, detok);

        var a = WavIo.ReadPcm(wav!);
        float[] mono = Resampler.ToMono16k(a.Interleaved, a.SampleRate, a.Channels);
        float[] feats = fe.Extract(mono, out int frames);
        dec.AcceptFeatures(feats, frames, 128);
        dec.InputFinished();
        string text = dec.GetResult().Text;

        Assert.False(string.IsNullOrWhiteSpace(text), "Qwen produced empty text");
        string? r = Environment.GetEnvironmentVariable("STT_QWEN_REF");
        if (!string.IsNullOrEmpty(r)) Assert.Contains(r.Trim(), text, StringComparison.OrdinalIgnoreCase);
    }
}
