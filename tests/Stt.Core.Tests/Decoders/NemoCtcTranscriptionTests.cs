using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Stt.Core.Audio;
using Stt.Core.Decoders;
using Stt.Core.Features;
using Stt.Core.Text;

namespace Stt.Core.Tests.Decoders;

/// <summary>
/// Real end-to-end test of the Family D (NeMo) front-end: a bundled speech WAV → NemoMelFrontend
/// (librosa mel + per-feature norm) → NeMo Conformer-CTC (ORT) → CTC greedy → text, asserting the
/// expected transcript. This proves NemoMelFrontend produces features a real NeMo model decodes
/// correctly. Skips unless the shim + a NeMo CTC model folder are present (env STT_NEMO_DIR).
/// </summary>
public class NemoCtcTranscriptionTests
{
    [SkippableTheory]
    [InlineData("0.wav", "nightfall")]
    [InlineData("1.wav", "consequence")]
    public void Transcribes_NeMo_Ctc(string wavName, string expectedSubstring)
    {
        string? dir = Environment.GetEnvironmentVariable("STT_NEMO_DIR");
        Skip.If(string.IsNullOrEmpty(dir) || !Directory.Exists(dir), "Set STT_NEMO_DIR to a NeMo CTC model folder.");
        Skip.IfNot(KaldiNativeFbankInterop.IsAvailable, "kaldi_native_fbank_shim not present.");

        string modelPath = Path.Combine(dir!, "model.onnx");
        string tokensPath = Path.Combine(dir!, "tokens.txt");
        Skip.IfNot(File.Exists(modelPath) && File.Exists(tokensPath), $"missing model.onnx/tokens.txt in {dir}.");

        using var session = new InferenceSession(modelPath);
        var tokens = TokenTable.Load(tokensPath);
        int blank = tokens.Count - 1; // NeMo: <blk> is the last token (id == vocab_size).

        // Family D features: librosa mel + per-feature norm, [T, 80].
        var wav = WavIo.ReadPcm(Path.Combine(dir!, "test_wavs", wavName));
        float[] mono = Resampler.ToMono16k(wav.Interleaved, wav.SampleRate, wav.Channels);
        using var fe = new NemoMelFrontend(80);
        float[] feats = fe.Extract(mono, out int frames);
        int mels = fe.FeatureDim;

        // NeMo expects audio_signal = [1, n_mels, T] (mel-major), so transpose from our [T, mels].
        var melMajor = new float[mels * frames];
        for (int t = 0; t < frames; t++)
            for (int d = 0; d < mels; d++)
                melMajor[d * frames + t] = feats[t * mels + d];

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("audio_signal", new DenseTensor<float>(melMajor, new[] { 1, mels, frames })),
            NamedOnnxValue.CreateFromTensor("length", new DenseTensor<long>(new[] { (long)frames }, new[] { 1 })),
        };

        using var results = session.Run(inputs);
        var logits = results.First(r => r.AsTensor<float>().Dimensions.Length == 3).AsTensor<float>();
        int outT = logits.Dimensions[1], vocab = logits.Dimensions[2];

        int[] ids = GreedyCtc.Decode(logits.ToArray(), outT, vocab, blank);
        string text = SentencePieceDetokenizer.Decode(ids.Select(tokens.Piece));

        File.AppendAllText(Path.Combine(Path.GetTempPath(), "stt_nemo.txt"), $"{wavName} => [{text}]\n");
        Assert.Contains(expectedSubstring, text, StringComparison.OrdinalIgnoreCase);
    }
}
