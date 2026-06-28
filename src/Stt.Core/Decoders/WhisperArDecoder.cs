using System.Text;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Stt.Abstractions.Decoders;

namespace Stt.Core.Decoders;

/// <summary>
/// Whisper decode configuration, read from the encoder ONNX metadata (sherpa-onnx export).
/// </summary>
public sealed record WhisperConfig(
    int Sot, int Eot, int Transcribe, int NoTimestamps,
    int NLayer, int NState, int NTextCtx, bool IsMultilingual, int[] LanguageTokens)
{
    public static WhisperConfig FromMetadata(IReadOnlyDictionary<string, string> m)
    {
        int I(string k, int def = 0) => m.TryGetValue(k, out var v) && int.TryParse(v, out var n) ? n : def;
        int[] langs = m.TryGetValue("all_language_tokens", out var lt) && !string.IsNullOrWhiteSpace(lt)
            ? lt.Split(',', StringSplitOptions.RemoveEmptyEntries).Select(int.Parse).ToArray()
            : Array.Empty<int>();
        return new WhisperConfig(
            Sot: I("sot", 50257), Eot: I("eot", 50256), Transcribe: I("transcribe", 50358),
            NoTimestamps: I("no_timestamps", 50362), NLayer: I("n_text_layer", 4),
            NState: I("n_text_state", 384), NTextCtx: I("n_text_ctx", 448),
            IsMultilingual: I("is_multilingual", 0) != 0, LanguageTokens: langs);
    }
}

/// <summary>
/// Offline autoregressive Whisper decoder (spec §8.5, Family C) over bare ONNX Runtime — encoder +
/// greedy decoder with a self-attention KV cache, no onnxruntime-genai. Buffers the Whisper log-mel
/// features (from <c>WhisperMelFrontend</c>), runs the encoder once to get the cross-attention K/V,
/// then greedily generates tokens until end-of-text. For multilingual models the spoken language is
/// detected from the first decoder step. Tokens are byte-level BPE (base64 pieces → bytes → UTF-8).
/// </summary>
public sealed class WhisperArDecoder : IAsrDecoder
{
    private readonly InferenceSession _encoder;
    private readonly InferenceSession _decoder;
    private readonly WhisperConfig _cfg;
    private readonly IReadOnlyDictionary<int, byte[]> _vocab;

    private readonly List<float> _featBuf = new();
    private int _featDim;
    private AsrResult _result = AsrResult.Empty;
    private bool _finished;

    public WhisperArDecoder(InferenceSession encoder, InferenceSession decoder,
                            WhisperConfig config, IReadOnlyDictionary<int, byte[]> vocab)
    {
        _encoder = encoder;
        _decoder = decoder;
        _cfg = config;
        _vocab = vocab;
    }

    public DecoderCapabilities Capabilities =>
        DecoderCapabilities.Offline | DecoderCapabilities.Multilingual;

    public void Reset()
    {
        _featBuf.Clear();
        _featDim = 0;
        _result = AsrResult.Empty;
        _finished = false;
    }

    public bool AcceptFeatures(ReadOnlySpan<float> features, int numFrames, int featDim)
    {
        if (_finished) return false;
        _featDim = featDim;
        for (int i = 0; i < features.Length; i++) _featBuf.Add(features[i]);
        return true;
    }

    public void InputFinished()
    {
        if (_finished) return;
        _finished = true;
        if (_featBuf.Count == 0 || _featDim == 0)
        {
            _result = new AsrResult(string.Empty, Array.Empty<int>(), Array.Empty<float>(), IsFinal: true);
            return;
        }

        int frames = _featBuf.Count / _featDim;
        // Encoder wants mel-major [1, n_mels, T]; our features are time-major [T, n_mels].
        var mel = new float[_featDim * frames];
        var buf = _featBuf;
        for (int t = 0; t < frames; t++)
            for (int d = 0; d < _featDim; d++)
                mel[d * frames + t] = buf[t * _featDim + d];

        using var enc = _encoder.Run(new[]
        {
            NamedOnnxValue.CreateFromTensor("mel", new DenseTensor<float>(mel, new[] { 1, _featDim, frames })),
        });
        float[] crossK = enc.First(r => r.Name == "n_layer_cross_k").AsTensor<float>().ToArray();
        float[] crossV = enc.First(r => r.Name == "n_layer_cross_v").AsTensor<float>().ToArray();
        int audioT = enc.First(r => r.Name == "n_layer_cross_k").AsTensor<float>().Dimensions[2];

        // Build the start-of-transcript prompt. Non-multilingual = [sot, no_timestamps];
        // multilingual = [sot, <detected lang>, transcribe, no_timestamps].
        var prompt = new List<long> { _cfg.Sot };
        if (_cfg.IsMultilingual && _cfg.LanguageTokens.Length > 0)
        {
            int lang = DetectLanguage(crossK, crossV, audioT);
            prompt.Add(lang);
            prompt.Add(_cfg.Transcribe);
        }
        prompt.Add(_cfg.NoTimestamps);

        var ids = Generate(prompt.ToArray(), crossK, crossV, audioT);

        var bytes = new List<byte>();
        foreach (int id in ids)
            if (_vocab.TryGetValue(id, out byte[]? b)) bytes.AddRange(b);
        string text = Encoding.UTF8.GetString(bytes.ToArray()).Trim();

        _result = new AsrResult(text, ids.ToArray(), Array.Empty<float>(), IsFinal: true);
    }

    /// <summary>One decoder step over <paramref name="tokens"/>; returns last-position logits + updated caches.</summary>
    private float[] DecodeStep(long[] tokens, float[] selfK, float[] selfV, float[] crossK, float[] crossV,
                               int audioT, long offset, out float[] outK, out float[] outV)
    {
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("tokens", new DenseTensor<long>(tokens, new[] { 1, tokens.Length })),
            NamedOnnxValue.CreateFromTensor("in_n_layer_self_k_cache", new DenseTensor<float>(selfK, new[] { _cfg.NLayer, 1, _cfg.NTextCtx, _cfg.NState })),
            NamedOnnxValue.CreateFromTensor("in_n_layer_self_v_cache", new DenseTensor<float>(selfV, new[] { _cfg.NLayer, 1, _cfg.NTextCtx, _cfg.NState })),
            NamedOnnxValue.CreateFromTensor("n_layer_cross_k", new DenseTensor<float>(crossK, new[] { _cfg.NLayer, 1, audioT, _cfg.NState })),
            NamedOnnxValue.CreateFromTensor("n_layer_cross_v", new DenseTensor<float>(crossV, new[] { _cfg.NLayer, 1, audioT, _cfg.NState })),
            NamedOnnxValue.CreateFromTensor("offset", new DenseTensor<long>(new[] { offset }, new[] { 1 })),
        };
        using var dec = _decoder.Run(inputs);
        var logits = dec.First(r => r.Name == "logits").AsTensor<float>();
        int n = logits.Dimensions[1], v = logits.Dimensions[2];
        var last = new float[v];
        for (int i = 0; i < v; i++) last[i] = logits[0, n - 1, i];
        outK = dec.First(r => r.Name == "out_n_layer_self_k_cache").AsTensor<float>().ToArray();
        outV = dec.First(r => r.Name == "out_n_layer_self_v_cache").AsTensor<float>().ToArray();
        return last;
    }

    private int DetectLanguage(float[] crossK, float[] crossV, int audioT)
    {
        float[] logits = DecodeStep(new[] { (long)_cfg.Sot }, NewCache(), NewCache(), crossK, crossV, audioT, 0, out _, out _);
        int best = _cfg.LanguageTokens[0]; float bestVal = float.NegativeInfinity;
        foreach (int lang in _cfg.LanguageTokens)
            if (lang < logits.Length && logits[lang] > bestVal) { bestVal = logits[lang]; best = lang; }
        return best;
    }

    private List<int> Generate(long[] prompt, float[] crossK, float[] crossV, int audioT)
    {
        float[] selfK = NewCache(), selfV = NewCache();
        long[] tokens = prompt;
        long offset = 0;
        var outTokens = new List<int>();

        for (int step = 0; step < _cfg.NTextCtx; step++)
        {
            float[] logits = DecodeStep(tokens, selfK, selfV, crossK, crossV, audioT, offset, out selfK, out selfV);
            int best = 0; float bestVal = float.NegativeInfinity;
            for (int v = 0; v < logits.Length; v++) if (logits[v] > bestVal) { bestVal = logits[v]; best = v; }
            offset += tokens.Length;
            if (best == _cfg.Eot) break;
            outTokens.Add(best);
            tokens = new long[] { best };
        }
        return outTokens;
    }

    private float[] NewCache() => new float[_cfg.NLayer * 1 * _cfg.NTextCtx * _cfg.NState];

    public bool IsEndpoint() => false;
    public AsrResult GetResult() => _result;

    public void Dispose()
    {
        _encoder.Dispose();
        _decoder.Dispose();
    }

    /// <summary>Load a sherpa-onnx Whisper tokens.txt: each line is <c>base64piece id</c>; special tokens skip.</summary>
    public static Dictionary<int, byte[]> LoadVocab(string tokensPath)
    {
        var map = new Dictionary<int, byte[]>();
        foreach (string line in File.ReadAllLines(tokensPath))
        {
            int sp = line.LastIndexOf(' ');
            if (sp <= 0) continue;
            if (!int.TryParse(line[(sp + 1)..], out int id)) continue;
            try { map[id] = Convert.FromBase64String(line[..sp]); } catch { /* special / non-base64 */ }
        }
        return map;
    }
}
