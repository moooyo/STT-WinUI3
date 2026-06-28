using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Stt.Abstractions.Audio;
using Stt.Abstractions.Vad;

namespace Stt.Core.Vad;

/// <summary>
/// Silero VAD (spec §5.1, §16) over ONNX Runtime. Runs the model per 512-sample window to get a
/// speech probability, then feeds it to a <see cref="VadSegmenter"/> for endpointing. Supports
/// both the v4 (separate h/c LSTM state) and v5 (combined state) export layouts, detected from
/// the model's input names. The model is user-supplied (D7); construction throws if absent.
/// </summary>
public sealed class SileroVad : IVad
{
    private readonly InferenceSession _session;
    private readonly VadOptions _o;
    private readonly VadSegmenter _segmenter;
    private readonly bool _isV5;
    private readonly long _sampleRate;

    // Recurrent state (v5: one [2,1,128] tensor; v4: h/c each [2,1,64]).
    private float[] _state = Array.Empty<float>();
    private float[] _h = Array.Empty<float>();
    private float[] _c = Array.Empty<float>();

    public SileroVad(string modelPath, VadOptions options, SessionOptions? sessionOptions = null)
    {
        if (!File.Exists(modelPath)) throw new FileNotFoundException("Silero VAD model not found.", modelPath);
        _session = sessionOptions is null ? new InferenceSession(modelPath)
                                          : new InferenceSession(modelPath, sessionOptions);
        _o = options;
        _segmenter = new VadSegmenter(options);
        _sampleRate = options.SampleRate;
        _isV5 = _session.InputMetadata.ContainsKey("state");
        Reset();
    }

    public void Reset()
    {
        _segmenter.Reset();
        if (_isV5) _state = new float[2 * 1 * 128];
        else { _h = new float[2 * 1 * 64]; _c = new float[2 * 1 * 64]; }
    }

    public void AcceptWaveform(ReadOnlySpan<float> window512)
    {
        float prob = RunModel(window512);
        _segmenter.AcceptWindow(prob, window512);
    }

    public bool TryDequeueSegment(out SpeechSegment seg) => _segmenter.TryDequeue(out seg);

    /// <summary>Force-close an open segment (called by the pipeline at end of capture).</summary>
    public void Flush() => _segmenter.Flush();

    private float RunModel(ReadOnlySpan<float> window)
    {
        var input = new DenseTensor<float>(window.ToArray(), new[] { 1, window.Length });
        var sr = new DenseTensor<long>(new[] { _sampleRate }, new[] { 1 });

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input", input),
            NamedOnnxValue.CreateFromTensor("sr", sr),
        };

        if (_isV5)
            inputs.Add(NamedOnnxValue.CreateFromTensor("state", new DenseTensor<float>(_state, new[] { 2, 1, 128 })));
        else
        {
            inputs.Add(NamedOnnxValue.CreateFromTensor("h", new DenseTensor<float>(_h, new[] { 2, 1, 64 })));
            inputs.Add(NamedOnnxValue.CreateFromTensor("c", new DenseTensor<float>(_c, new[] { 2, 1, 64 })));
        }

        using var results = _session.Run(inputs);
        float prob = 0f;
        foreach (var r in results)
        {
            switch (r.Name)
            {
                case "output": prob = r.AsTensor<float>().ToArray()[0]; break;
                case "stateN": _state = r.AsTensor<float>().ToArray(); break;
                case "hn": _h = r.AsTensor<float>().ToArray(); break;
                case "cn": _c = r.AsTensor<float>().ToArray(); break;
            }
        }
        return prob;
    }

    public void Dispose() => _session.Dispose();
}
