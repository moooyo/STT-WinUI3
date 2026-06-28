namespace Stt.Core.Decoders;

/// <summary>
/// Seam over a single-shot ONNX forward pass, so decode logic (CTC collapse, detokenize, tag
/// strip) is unit testable with canned logits and the real ORT runner is swapped in at runtime.
/// </summary>
public interface IModelRunner
{
    /// <summary>
    /// Run the model on row-major <c>[numFrames, featDim]</c> features and return row-major
    /// <c>[outFrames, vocab]</c> logits/log-probs.
    /// </summary>
    float[] Run(float[] features, int numFrames, int featDim, out int outFrames, out int vocab);
}
