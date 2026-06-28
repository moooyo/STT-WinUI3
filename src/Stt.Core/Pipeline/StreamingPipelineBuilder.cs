using Microsoft.ML.OnnxRuntime;
using Stt.Abstractions.Decoders;
using Stt.Abstractions.Ep;
using Stt.Abstractions.Features;
using Stt.Abstractions.Models;
using Stt.Core.Decoders;
using Stt.Core.Ep;
using Stt.Core.Features;
using Stt.Core.Models;
using Stt.Core.Text;

namespace Stt.Core.Pipeline;

/// <summary>The streaming (first-pass) decode chain: a fbank front-end + transducer decoder.</summary>
public sealed record StreamingChain(IFeatureFrontend Frontend, IAsrDecoder Decoder) : IDisposable
{
    public void Dispose()
    {
        (Frontend as IDisposable)?.Dispose();
        Decoder.Dispose();
    }
}

/// <summary>
/// Wires a streaming Zipformer transducer chain from a model folder (spec §8.2, D3): loads the
/// encoder/decoder/joiner with the selected EP, reads the state geometry from encoder metadata,
/// and builds an <see cref="OrtStreamingSession"/> + <see cref="TransducerDecoder"/> with a
/// Family-A kaldi-fbank front-end. Requires the real model files + ORT native runtime.
/// </summary>
public static class StreamingPipelineBuilder
{
    public static StreamingChain BuildStreaming(
        ModelManifest manifest,
        IExecutionProviderSelector epSelector,
        EpPreference epPreference,
        double minTrailingSilenceSeconds = 0.8)
    {
        if (manifest.FolderPath is null)
            throw new InvalidOperationException("Manifest has no folder path; import it via the registry first.");

        string Path3(string? f, string what) => f is not null
            ? Path.Combine(manifest.FolderPath, f)
            : throw new InvalidOperationException($"Streaming model '{manifest.Id}' is missing its {what} file.");

        string encPath = Path3(manifest.Files.Encoder, "encoder");
        string decPath = Path3(manifest.Files.Decoder, "decoder");
        string joiPath = Path3(manifest.Files.Joiner, "joiner");
        string tokPath = Path3(manifest.Files.Tokens, "tokens");

        string hash = $"{manifest.Id}-{new FileInfo(encPath).Length}";
        SessionOptions opts = epSelector.BuildSessionOptions(epPreference, hash);

        var encoder = new InferenceSession(encPath, opts);
        var decoder = new InferenceSession(decPath, opts);
        var joiner = new InferenceSession(joiPath, opts);

        var geo = ZipformerGeometry.FromMetadata(ModelMetadataReader.FromSession(encoder).Metadata);
        var session = new OrtStreamingSession(encoder, decoder, joiner, geo);
        var tokens = TokenTable.Load(tokPath);

        var transducer = new TransducerDecoder(session, tokens, endpointSilenceSeconds: minTrailingSilenceSeconds);

        int bins = manifest.Feature.FeatureDim > 0 ? manifest.Feature.FeatureDim : 80;
        var frontend = new KaldiFbankFrontend(FbankOptions.FamilyA(bins));

        return new StreamingChain(frontend, transducer);
    }
}
