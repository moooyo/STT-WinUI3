using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Stt.Abstractions.Audio;
using Stt.Abstractions.Vad;

namespace Stt.Core.Vad;

/// <summary>
/// Silero VAD (spec §5.1, §16) over ONNX Runtime. Runs the model per 512-sample window to get a
/// speech probability, then feeds it to a <see cref="VadSegmenter"/> for endpointing.
/// </summary>
/// <remarks>
/// Input/output names differ across silero exports — the waveform is <c>input</c> or <c>x</c>; a
/// sample-rate input is present on some exports and absent on others; the recurrent state is one
/// combined <c>state</c> tensor (v5) or separate <c>h</c>/<c>c</c> (v4); the probability output is
/// <c>output</c> or <c>prob</c>; next-state is <c>stateN</c> or <c>new_h</c>/<c>new_c</c>. So the
/// I/O is discovered from the model metadata <b>by role</b> (dtype + rank) rather than hard-coded,
/// matching the engine's fail-loud, metadata-driven design. The model is user-supplied (D7).
/// </remarks>
public sealed class SileroVad : IVad
{
    private readonly InferenceSession _session;
    private readonly VadSegmenter _segmenter;
    private readonly long _sampleRate;

    private readonly string _waveInput;                          // waveform input (rank ≤ 2 float)
    private readonly string? _srInput;                           // optional int64 sample-rate input
    private readonly string _probOutput;                         // smallest float output ([1,1])
    private readonly (string Name, int[] Shape)[] _stateInputs;  // recurrent-state inputs
    private readonly string[] _stateOutputs;                     // next-state, positional with inputs
    private readonly float[][] _state;

    public SileroVad(string modelPath, VadOptions options, SessionOptions? sessionOptions = null)
    {
        if (!File.Exists(modelPath)) throw new FileNotFoundException("Silero VAD model not found.", modelPath);
        _session = sessionOptions is null ? new InferenceSession(modelPath)
                                          : new InferenceSession(modelPath, sessionOptions);
        _segmenter = new VadSegmenter(options);
        _sampleRate = options.SampleRate;

        // Inputs: an int64 input (if present) is the sample rate; among the float inputs the rank-≤2
        // one is the waveform and the remaining (rank-3) ones are recurrent state.
        string? srName = null;
        var floatInputs = new List<KeyValuePair<string, NodeMetadata>>();
        foreach (var kv in _session.InputMetadata)
        {
            if (kv.Value.ElementDataType == TensorElementType.Int64) srName = kv.Key;
            else floatInputs.Add(kv);
        }
        if (floatInputs.Count == 0)
            throw new InvalidOperationException("Silero VAD model has no float inputs — not a recognized VAD export.");

        var wave = floatInputs.FirstOrDefault(kv => kv.Value.Dimensions.Length <= 2);
        _waveInput = wave.Key ?? floatInputs[0].Key;
        _srInput = srName;
        _stateInputs = floatInputs.Where(kv => kv.Key != _waveInput)
            .Select(kv => (kv.Key, Concrete(kv.Value.Dimensions))).ToArray();

        // Outputs: the smallest-volume float output is the probability ([1,1]); the rest are
        // next-state, mapped positionally onto the state inputs (h→new_h, c→new_c, state→stateN).
        var outs = _session.OutputMetadata.ToList();
        _probOutput = outs.OrderBy(kv => Volume(kv.Value.Dimensions)).First().Key;
        _stateOutputs = outs.Where(kv => kv.Key != _probOutput).Select(kv => kv.Key).ToArray();

        _state = new float[_stateInputs.Length][];
        Reset();
    }

    public void Reset()
    {
        _segmenter.Reset();
        for (int i = 0; i < _stateInputs.Length; i++)
        {
            int n = 1;
            foreach (int d in _stateInputs[i].Shape) n *= d;
            _state[i] = new float[n];
        }
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
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_waveInput, new DenseTensor<float>(window.ToArray(), new[] { 1, window.Length })),
        };
        if (_srInput is not null)
            inputs.Add(NamedOnnxValue.CreateFromTensor(_srInput, new DenseTensor<long>(new[] { _sampleRate }, new[] { 1 })));
        for (int i = 0; i < _stateInputs.Length; i++)
            inputs.Add(NamedOnnxValue.CreateFromTensor(_stateInputs[i].Name, new DenseTensor<float>(_state[i], _stateInputs[i].Shape)));

        using var results = _session.Run(inputs);
        var byName = results.ToDictionary(r => r.Name, r => r);
        float prob = byName[_probOutput].AsTensor<float>().ToArray()[0];
        for (int i = 0; i < _stateOutputs.Length && i < _state.Length; i++)
            _state[i] = byName[_stateOutputs[i]].AsTensor<float>().ToArray();
        return prob;
    }

    public void Dispose() => _session.Dispose();

    /// <summary>Replace any dynamic (≤0) dim with 1 to get a concrete recurrent-state shape.</summary>
    private static int[] Concrete(int[] dims)
    {
        var s = new int[dims.Length];
        for (int i = 0; i < dims.Length; i++) s[i] = dims[i] > 0 ? dims[i] : 1;
        return s;
    }

    private static long Volume(int[] dims)
    {
        long v = 1;
        foreach (int d in dims) v *= d > 0 ? d : 1;
        return v;
    }
}
