using System.Text;
using Microsoft.ML.OnnxRuntimeGenAI;
using Stt.Abstractions.Decoders;
using Stt.Core.Audio;

namespace Stt.Plugins.WhisperGenAi;

/// <summary>
/// Optional Whisper autoregressive decoder over onnxruntime-genai (spec §8.5, D-reserved). genai
/// owns the AR loop + KV cache + beam internally. <b>Not in the v1 core pipeline:</b> it carries
/// its own ORT/EP (does not inherit the unified Windows ML EP selection), is offline/30 s-chunked,
/// and its C# API is preview. This class exists as the drop-in for the <see cref="DecoderType.Ar"/>
/// capability so a future pipeline can select it like any other <see cref="IAsrDecoder"/>.
/// </summary>
/// <remarks>
/// Whisper consumes audio (mel is computed inside the model), so this decoder buffers raw 16 kHz
/// mono PCM via <see cref="AcceptFeatures"/> (treat the "features" as pass-through audio) and runs
/// generation on <see cref="InputFinished"/>.
/// </remarks>
public sealed class WhisperGenAiDecoder : IAsrDecoder
{
    private readonly string _modelDir;
    private readonly string _prompt;
    private readonly int _maxLength;
    private readonly List<float> _audio = new();
    private AsrResult _result = AsrResult.Empty;
    private bool _finished;

    public WhisperGenAiDecoder(string modelDir, string? prompt = null, int maxLength = 448)
    {
        _modelDir = modelDir;
        _prompt = prompt ?? "<|startoftranscript|>";
        _maxLength = maxLength;
    }

    public DecoderCapabilities Capabilities =>
        DecoderCapabilities.Offline | DecoderCapabilities.Multilingual;

    public void Reset()
    {
        _audio.Clear();
        _result = AsrResult.Empty;
        _finished = false;
    }

    /// <summary>Buffer pass-through 16 kHz mono PCM (Whisper computes mel internally).</summary>
    public bool AcceptFeatures(ReadOnlySpan<float> audio, int numFrames, int featDim)
    {
        if (_finished) return false;
        for (int i = 0; i < audio.Length; i++) _audio.Add(audio[i]);
        return true;
    }

    public void InputFinished()
    {
        if (_finished) return;
        _finished = true;
        if (_audio.Count == 0)
        {
            _result = new AsrResult(string.Empty, Array.Empty<int>(), Array.Empty<float>(), IsFinal: true);
            return;
        }

        string wavPath = Path.Combine(Path.GetTempPath(), $"whisper_{Guid.NewGuid():N}.wav");
        WavIo.WritePcm16(wavPath, _audio.ToArray(), 16000);
        try
        {
            _result = new AsrResult(Generate(wavPath), Array.Empty<int>(), Array.Empty<float>(), IsFinal: true);
        }
        finally
        {
            try { File.Delete(wavPath); } catch { /* best effort */ }
        }
    }

    private string Generate(string wavPath)
    {
        using var model = new Model(_modelDir);
        using var processor = new MultiModalProcessor(model);
        using var stream = processor.CreateStream();

        using var audios = Audios.Load(new[] { wavPath });
        // Whisper is audio-only (no images). genai computes the log-mel inside the model.
        using var input = processor.ProcessImagesAndAudios(_prompt, null, audios);

        using var generatorParams = new GeneratorParams(model);
        generatorParams.SetSearchOption("max_length", _maxLength);
        generatorParams.SetInputs(input);

        using var generator = new Generator(model, generatorParams);
        var sb = new StringBuilder();
        while (!generator.IsDone())
        {
            generator.GenerateNextToken();
            ReadOnlySpan<int> seq = generator.GetSequence(0);
            if (seq.Length > 0) sb.Append(stream.Decode(seq[seq.Length - 1]));
        }
        return sb.ToString().Trim();
    }

    public bool IsEndpoint() => false;
    public AsrResult GetResult() => _result;
    public void Dispose() { }
}
