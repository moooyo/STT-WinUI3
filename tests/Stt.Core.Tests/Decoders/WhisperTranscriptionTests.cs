using System.Text;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Stt.Core.Audio;
using Stt.Core.Features;

namespace Stt.Core.Tests.Decoders;

/// <summary>
/// Real end-to-end test of Family C (Whisper) front-end: a bundled speech WAV → WhisperMelFrontend
/// → Whisper encoder → greedy autoregressive decoder (KV cache) → text, asserting the expected
/// transcript. Proves WhisperMelFrontend feeds a real Whisper model correctly. Skips unless the shim
/// and a sherpa-onnx Whisper model folder are present (env STT_WHISPER_DIR, with
/// <c>{prefix}-encoder.onnx</c>/<c>-decoder.onnx</c>/<c>-tokens.txt</c>).
/// </summary>
public class WhisperTranscriptionTests
{
    // tiny.en metadata (sherpa-onnx export).
    private const int Sot = 50257, Eot = 50256, NoTimestamps = 50362, NLayer = 4, NState = 384, NTextCtx = 448;

    [SkippableTheory]
    [InlineData("0.wav", "nightfall")]
    [InlineData("1.wav", "consequence")]
    public void Transcribes_Whisper(string wavName, string expectedSubstring)
    {
        string? dir = Environment.GetEnvironmentVariable("STT_WHISPER_DIR");
        Skip.If(string.IsNullOrEmpty(dir) || !Directory.Exists(dir), "Set STT_WHISPER_DIR to a sherpa-onnx Whisper model folder.");
        Skip.IfNot(KaldiNativeFbankInterop.IsAvailable, "kaldi_native_fbank_shim not present.");

        string prefix = Directory.GetFiles(dir!, "*-encoder.onnx").FirstOrDefault()?[..^"-encoder.onnx".Length]
            ?? throw new FileNotFoundException("no *-encoder.onnx in STT_WHISPER_DIR");
        using var encoder = new InferenceSession(prefix + "-encoder.onnx");
        using var decoder = new InferenceSession(prefix + "-decoder.onnx");
        var vocab = LoadBase64Tokens(prefix + "-tokens.txt");

        // Family C features → mel-major [1, 80, 3000].
        var wav = WavIo.ReadPcm(Path.Combine(dir!, "test_wavs", wavName));
        float[] mono = Resampler.ToMono16k(wav.Interleaved, wav.SampleRate, wav.Channels);
        using var fe = new WhisperMelFrontend(80);
        float[] feats = fe.Extract(mono, out int frames);   // [3000, 80]
        var mel = new float[80 * frames];
        for (int t = 0; t < frames; t++)
            for (int d = 0; d < 80; d++) mel[d * frames + t] = feats[t * 80 + d];

        // Encoder → cross-attention K/V.
        using var enc = encoder.Run(new[]
        {
            NamedOnnxValue.CreateFromTensor("mel", new DenseTensor<float>(mel, new[] { 1, 80, frames })),
        });
        var crossK = enc.First(r => r.Name == "n_layer_cross_k").AsTensor<float>();
        var crossV = enc.First(r => r.Name == "n_layer_cross_v").AsTensor<float>();
        int audioT = crossK.Dimensions[2];
        float[] crossKArr = crossK.ToArray(), crossVArr = crossV.ToArray();

        // Greedy AR decode.
        var selfK = new float[NLayer * 1 * NTextCtx * NState];
        var selfV = new float[NLayer * 1 * NTextCtx * NState];
        long[] tokens = { Sot, NoTimestamps };
        long offset = 0;
        var outTokens = new List<int>();

        for (int step = 0; step < NTextCtx; step++)
        {
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("tokens", new DenseTensor<long>(tokens, new[] { 1, tokens.Length })),
                NamedOnnxValue.CreateFromTensor("in_n_layer_self_k_cache", new DenseTensor<float>(selfK, new[] { NLayer, 1, NTextCtx, NState })),
                NamedOnnxValue.CreateFromTensor("in_n_layer_self_v_cache", new DenseTensor<float>(selfV, new[] { NLayer, 1, NTextCtx, NState })),
                NamedOnnxValue.CreateFromTensor("n_layer_cross_k", new DenseTensor<float>(crossKArr, new[] { NLayer, 1, audioT, NState })),
                NamedOnnxValue.CreateFromTensor("n_layer_cross_v", new DenseTensor<float>(crossVArr, new[] { NLayer, 1, audioT, NState })),
                NamedOnnxValue.CreateFromTensor("offset", new DenseTensor<long>(new[] { offset }, new[] { 1 })),
            };
            using var dec = decoder.Run(inputs);
            var logits = dec.First(r => r.Name == "logits").AsTensor<float>();
            int n = logits.Dimensions[1], V = logits.Dimensions[2];
            // argmax over the last position's logits.
            int baseIdx = (n - 1) * V, best = 0; float bestVal = float.NegativeInfinity;
            for (int v = 0; v < V; v++) { float x = logits[0, n - 1, v]; if (x > bestVal) { bestVal = x; best = v; } }

            selfK = dec.First(r => r.Name == "out_n_layer_self_k_cache").AsTensor<float>().ToArray();
            selfV = dec.First(r => r.Name == "out_n_layer_self_v_cache").AsTensor<float>().ToArray();
            offset += tokens.Length;

            if (best == Eot) break;
            outTokens.Add(best);
            tokens = new long[] { best };
        }

        // Detokenize byte-level BPE: base64 piece → bytes → UTF-8.
        var bytes = new List<byte>();
        foreach (int id in outTokens)
            if (vocab.TryGetValue(id, out byte[]? b)) bytes.AddRange(b);
        string text = Encoding.UTF8.GetString(bytes.ToArray()).Trim();

        File.AppendAllText(Path.Combine(Path.GetTempPath(), "stt_whisper.txt"), $"{wavName} => [{text}]\n");
        Assert.Contains(expectedSubstring, text, StringComparison.OrdinalIgnoreCase);
    }

    private static Dictionary<int, byte[]> LoadBase64Tokens(string path)
    {
        var map = new Dictionary<int, byte[]>();
        foreach (string line in File.ReadAllLines(path))
        {
            int sp = line.LastIndexOf(' ');
            if (sp <= 0) continue;
            string piece = line[..sp];
            if (!int.TryParse(line[(sp + 1)..], out int id)) continue;
            try { map[id] = Convert.FromBase64String(piece); } catch { /* special token */ }
        }
        return map;
    }
}
