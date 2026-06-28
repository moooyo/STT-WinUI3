using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Stt.Core.Decoders;

/// <summary>
/// Drives the three streaming Zipformer transducer graphs (encoder + decoder/predictor + joiner)
/// over ORT, carrying the encoder state tensors across chunks (spec §8.2, §8.6). State tensors are
/// initialized from the metadata geometry (<see cref="EncoderStateFactory"/>) and threaded back in
/// after each encoder run.
/// </summary>
/// <remarks>
/// State inputs/outputs are mapped positionally in <see cref="EncoderStateFactory.BuildSpecs"/>
/// order: the encoder's non-feature inputs are the current states, and its outputs after
/// <c>encoder_out</c> are the next states. This ordering — and the per-chunk numerics — must be
/// validated against sherpa-onnx for the specific model before the decoder is trusted (Task 1.3
/// alignment test). The functional path here is correct; the IO-binding + state double-buffering
/// optimization (spec §8.6) is a perf refinement layered on top later.
/// </remarks>
public sealed class OrtStreamingSession : IDisposable
{
    private readonly InferenceSession _encoder;
    private readonly InferenceSession _decoder;
    private readonly InferenceSession _joiner;
    private readonly ZipformerGeometry _geo;

    private readonly string _featureInput;
    private readonly string[] _stateInputs;   // encoder non-feature inputs, in spec order
    private readonly string _encoderOutput;
    private readonly string[] _stateOutputs;  // encoder outputs after encoder_out, in spec order

    private readonly List<StateSpec> _stateSpecs;
    private readonly Array[] _stateData;       // float[] or long[] per state

    public OrtStreamingSession(InferenceSession encoder, InferenceSession decoder, InferenceSession joiner, ZipformerGeometry geometry)
    {
        _encoder = encoder;
        _decoder = decoder;
        _joiner = joiner;
        _geo = geometry;

        _stateSpecs = EncoderStateFactory.BuildSpecs(geometry);

        // Identify the feature input (rank-3) vs the state inputs (everything else, in order).
        _featureInput = encoder.InputMetadata.FirstOrDefault(kv => kv.Value.Dimensions.Length == 3).Key
            ?? encoder.InputMetadata.Keys.First();
        _stateInputs = encoder.InputMetadata.Keys.Where(k => k != _featureInput).ToArray();

        _encoderOutput = encoder.OutputMetadata.FirstOrDefault(kv => kv.Value.Dimensions.Length == 3).Key
            ?? encoder.OutputMetadata.Keys.First();
        _stateOutputs = encoder.OutputMetadata.Keys.Where(k => k != _encoderOutput).ToArray();

        _stateData = new Array[_stateSpecs.Count];
        ResetStates();
    }

    public ZipformerGeometry Geometry => _geo;

    public void ResetStates()
    {
        for (int i = 0; i < _stateSpecs.Count; i++)
        {
            long count = 1;
            foreach (long d in _stateSpecs[i].Shape) count *= d;
            _stateData[i] = _stateSpecs[i].DType == StateDType.Float32 ? new float[count] : new long[count];
        }
    }

    /// <summary>Run the encoder on a feature chunk; returns the encoder output frames [outT][encDim].</summary>
    public float[][] Encode(float[] chunkFeatures, int frames, int featDim)
    {
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_featureInput, new DenseTensor<float>(chunkFeatures, new[] { 1, frames, featDim })),
        };
        for (int i = 0; i < _stateInputs.Length && i < _stateSpecs.Count; i++)
            inputs.Add(MakeStateValue(_stateInputs[i], _stateSpecs[i], _stateData[i]));

        using var results = _encoder.Run(inputs);
        var byName = results.ToDictionary(r => r.Name, r => r);

        // Carry new states back (positional with the state inputs).
        for (int i = 0; i < _stateOutputs.Length && i < _stateSpecs.Count; i++)
        {
            var outVal = byName[_stateOutputs[i]];
            _stateData[i] = _stateSpecs[i].DType == StateDType.Float32
                ? outVal.AsTensor<float>().ToArray()
                : outVal.AsTensor<long>().ToArray();
        }

        var enc = byName[_encoderOutput].AsTensor<float>();
        int outT = enc.Dimensions[1];
        int encDim = enc.Dimensions[2];
        var encArr = enc.ToArray();
        var rows = new float[outT][];
        for (int t = 0; t < outT; t++)
        {
            rows[t] = new float[encDim];
            Array.Copy(encArr, t * encDim, rows[t], 0, encDim);
        }
        return rows;
    }

    /// <summary>Run the predictor on the last context tokens; returns the flattened decoder_out.</summary>
    public float[] Predict(int[] context)
    {
        var tokens = new long[context.Length];
        for (int i = 0; i < context.Length; i++) tokens[i] = context[i];

        string inName = _decoder.InputMetadata.Keys.First();
        var inputs = new[] { NamedOnnxValue.CreateFromTensor(inName, new DenseTensor<long>(tokens, new[] { 1, context.Length })) };
        using var results = _decoder.Run(inputs);
        return results.First().AsTensor<float>().ToArray();
    }

    /// <summary>Run the joiner on an encoder frame + decoder_out; returns the vocab logits.</summary>
    public float[] Join(float[] encoderFrame, float[] decoderOut)
    {
        var names = _joiner.InputMetadata.Keys.ToArray();
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(names[0], new DenseTensor<float>(encoderFrame, new[] { 1, encoderFrame.Length })),
            NamedOnnxValue.CreateFromTensor(names[1], new DenseTensor<float>(decoderOut, new[] { 1, decoderOut.Length })),
        };
        using var results = _joiner.Run(inputs);
        return results.First().AsTensor<float>().ToArray();
    }

    private static NamedOnnxValue MakeStateValue(string name, StateSpec spec, Array data)
    {
        int[] shape = Array.ConvertAll(spec.Shape, x => (int)x);
        return spec.DType == StateDType.Float32
            ? NamedOnnxValue.CreateFromTensor(name, new DenseTensor<float>((float[])data, shape))
            : NamedOnnxValue.CreateFromTensor(name, new DenseTensor<long>((long[])data, shape));
    }

    public void Dispose()
    {
        _encoder.Dispose();
        _decoder.Dispose();
        _joiner.Dispose();
    }
}
