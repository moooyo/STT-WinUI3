using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Stt.Abstractions.Decoders;

namespace Stt.Core.Decoders;

/// <summary>
/// Qwen3-ASR decoder (Family C log-mel, 128-bin) over three ONNX graphs: encoder (mel → audio
/// features), decoder_init (prefill with audio-pad prompt) and decoder_step (autoregressive). The
/// LLM decoder is a "prefix-LM": the prompt is im_start/system/user + N audio_pad placeholders that
/// decoder_init replaces with encoder features at audio_offset, then greedy AR with a 28-layer KV
/// cache. Position ids are a plain range (no mRoPE on the ONNX path). Token ids/sequence match the
/// reference andrewleech/qwen3-asr export. Autoregressive ⇒ CPU / DirectML (not NPU/TensorRT).
/// </summary>
public sealed class QwenAsrDecoder : IAsrDecoder
{
    // Special token ids (shared across Qwen3-ASR sizes).
    private const int ImStart = 151644, ImEnd = 151645, Eot = 151643;
    private const int AudioStart = 151669, AudioEnd = 151670, AudioPad = 151676, Newline = 198;
    private const int SystemTok = 9125, UserTok = 882, AssistantTok = 77091;
    private static readonly int[] Eos = { Eot, ImEnd };

    private readonly InferenceSession _encoder, _init, _step;
    private readonly float[] _embed;     // [vocab, hidden] row-major, fp32
    private readonly int _hidden;
    private readonly Func<IReadOnlyList<int>, string> _detok;
    private readonly int _maxTokens;

    private int _featDim;
    private readonly List<float> _mel = new();
    private bool _finished;
    private string _text = "";

    public QwenAsrDecoder(InferenceSession encoder, InferenceSession init, InferenceSession step,
        float[] embedTokens, int hidden, Func<IReadOnlyList<int>, string> detok, int maxTokens = 256)
    {
        _encoder = encoder; _init = init; _step = step;
        _embed = embedTokens; _hidden = hidden; _detok = detok; _maxTokens = maxTokens;
    }

    public DecoderCapabilities Capabilities => DecoderCapabilities.Offline | DecoderCapabilities.Multilingual;

    public void Reset() { _mel.Clear(); _finished = false; _text = ""; _featDim = 0; }

    public bool AcceptFeatures(ReadOnlySpan<float> features, int numFrames, int featDim)
    {
        _featDim = featDim;
        for (int i = 0; i < features.Length; i++) _mel.Add(features[i]);
        return false; // offline: emit on InputFinished
    }

    public void InputFinished()
    {
        if (_finished) return;
        _finished = true;
        if (_mel.Count == 0 || _featDim <= 0) return;

        int frames = _mel.Count / _featDim;
        float[] audio = Encode(frames);                 // [T, 1024]
        int audioLen = audio.Length / _hidden;
        var prompt = BuildPrompt(audioLen);
        var tokens = Generate(prompt, audio, audioLen);
        _text = _detok(tokens);
    }

    /// <summary>mel rows [frames,128] → encoder feed [1,128,frames]; out audio_features [T,1024].</summary>
    private float[] Encode(int frames)
    {
        var mel = new float[frames * _featDim];
        for (int t = 0; t < frames; t++)
            for (int c = 0; c < _featDim; c++)
                mel[c * frames + t] = _mel[t * _featDim + c];  // transpose to [128, frames]

        string mIn = _encoder.InputMetadata.Keys.First();
        var inputs = new[] { NamedOnnxValue.CreateFromTensor(mIn, new DenseTensor<float>(mel, new[] { 1, _featDim, frames })) };
        using var r = _encoder.Run(inputs);
        return r.First().AsTensor<float>().ToArray();
    }

    private static List<int> BuildPrompt(int audioTokens)
    {
        var ids = new List<int> { ImStart, SystemTok, Newline, ImEnd, Newline,
                                  ImStart, UserTok, Newline, AudioStart };
        for (int i = 0; i < audioTokens; i++) ids.Add(AudioPad);
        ids.Add(AudioEnd); ids.Add(ImEnd); ids.Add(Newline);
        ids.Add(ImStart); ids.Add(AssistantTok); ids.Add(Newline);
        return ids;
    }

    private List<int> Generate(List<int> prompt, float[] audio, int audioLen)
    {
        int L = prompt.Count;
        int audioOffset = prompt.IndexOf(AudioPad);
        var ids = prompt.Select(x => (long)x).ToArray();
        var pos = new long[L]; for (int i = 0; i < L; i++) pos[i] = i;

        using var r = _init.Run(new[]
        {
            NamedOnnxValue.CreateFromTensor("input_ids", new DenseTensor<long>(ids, new[] { 1, L })),
            NamedOnnxValue.CreateFromTensor("position_ids", new DenseTensor<long>(pos, new[] { 1, L })),
            NamedOnnxValue.CreateFromTensor("audio_features", new DenseTensor<float>(audio, new[] { 1, audioLen, _hidden })),
            NamedOnnxValue.CreateFromTensor("audio_offset", new DenseTensor<long>(new long[] { audioOffset }, new[] { 1 })),
        });
        var byName = r.ToDictionary(x => x.Name, x => x.AsTensor<float>().ToArray());
        int next = ArgMaxLast(byName["logits"], L);
        var keys = byName["present_keys"]; var vals = byName["present_values"];
        var outTokens = new List<int>();
        if (Array.IndexOf(Eos, next) >= 0) return outTokens;
        outTokens.Add(next);

        int p = L;
        for (int s = 0; s < _maxTokens - 1; s++)
        {
            var emb = new float[_hidden];
            Array.Copy(_embed, (long)next * _hidden, emb, 0, _hidden);
            using var sr = _step.Run(new[]
            {
                NamedOnnxValue.CreateFromTensor("input_embeds", new DenseTensor<float>(emb, new[] { 1, 1, _hidden })),
                NamedOnnxValue.CreateFromTensor("position_ids", new DenseTensor<long>(new long[] { p }, new[] { 1, 1 })),
                MakeKv("past_keys", keys), MakeKv("past_values", vals),
            });
            var o = sr.ToDictionary(x => x.Name, x => x.AsTensor<float>());
            next = ArgMaxLast(o["logits"].ToArray(), 1);
            keys = o["present_keys"].ToArray(); vals = o["present_values"].ToArray();
            p++;
            if (Array.IndexOf(Eos, next) >= 0) break;
            outTokens.Add(next);
        }
        return outTokens;
    }

    // present K/V shape [28,1,8,seq,128]; reshape per step.
    private static NamedOnnxValue MakeKv(string name, float[] data)
    {
        int seq = data.Length / (28 * 8 * 128);
        return NamedOnnxValue.CreateFromTensor(name, new DenseTensor<float>(data, new[] { 28, 1, 8, seq, 128 }));
    }

    private static int ArgMaxLast(float[] logits, int seq)
    {
        int vocab = logits.Length / seq, off = (seq - 1) * vocab, best = 0; float bv = float.NegativeInfinity;
        for (int i = 0; i < vocab; i++) { float v = logits[off + i]; if (v > bv) { bv = v; best = i; } }
        return best;
    }

    public bool IsEndpoint() => true;
    public AsrResult GetResult() => new(_text, Array.Empty<int>(), Array.Empty<float>(), IsFinal: _finished);
    public void Dispose() { _encoder.Dispose(); _init.Dispose(); _step.Dispose(); }

    /// <summary>Load embed_tokens.bin (float16 [vocab,hidden]) as fp32 for step lookups.</summary>
    public static float[] LoadEmbedTokens(string path)
    {
        byte[] raw = File.ReadAllBytes(path);
        int n = raw.Length / 2;
        var f = new float[n];
        for (int i = 0; i < n; i++) f[i] = (float)BitConverter.ToHalf(raw, i * 2);
        return f;
    }

    /// <summary>
    /// Build a detokenizer from vocab.json (GPT-2 byte-level BPE, like Qwen). Maps ids→token strings,
    /// reverses the byte-level alphabet to raw bytes, UTF-8 decodes, and strips the "language X&lt;asr_text&gt;"
    /// transcription prefix the model emits.
    /// </summary>
    public static Func<IReadOnlyList<int>, string> LoadDetok(string vocabJsonPath)
    {
        using var doc = System.Text.Json.JsonDocument.Parse(File.ReadAllText(vocabJsonPath));
        var idToTok = new Dictionary<int, string>();
        foreach (var p in doc.RootElement.EnumerateObject())
            if (p.Value.TryGetInt32(out int id)) idToTok[id] = p.Name;
        var byteDecoder = ByteLevelDecoder();

        return ids =>
        {
            var bytes = new List<byte>();
            foreach (int id in ids)
                if (idToTok.TryGetValue(id, out var t))
                    foreach (char c in t) if (byteDecoder.TryGetValue(c, out byte b)) bytes.Add(b);
            string s = System.Text.Encoding.UTF8.GetString(bytes.ToArray());
            int idx = s.IndexOf("<asr_text>", StringComparison.Ordinal);
            if (idx >= 0) s = s[(idx + "<asr_text>".Length)..];
            return s.Trim();
        };
    }

    private static Dictionary<char, byte> ByteLevelDecoder()
    {
        var map = new Dictionary<char, byte>();
        var bs = new List<int>();
        for (int i = '!'; i <= '~'; i++) bs.Add(i);
        for (int i = 0xA1; i <= 0xAC; i++) bs.Add(i);
        for (int i = 0xAE; i <= 0xFF; i++) bs.Add(i);
        int n = 0;
        for (int b = 0; b < 256; b++)
            if (!bs.Contains(b)) { bs.Add(b); map[(char)(256 + n)] = (byte)b; n++; }
        foreach (int b in bs) if (b < 256 && !map.ContainsValue((byte)b)) map[(char)b] = (byte)b;
        return map;
    }
}
