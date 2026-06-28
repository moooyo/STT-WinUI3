namespace Stt.Abstractions.Decoders;

/// <summary>Decoding family of a model (spec §5.2).</summary>
public enum DecoderType
{
    /// <summary>RNN-T / Zipformer transducer (encoder + decoder/predictor + joiner).</summary>
    Transducer,

    /// <summary>Connectionist Temporal Classification: single encoder → log-probs.</summary>
    Ctc,

    /// <summary>Non-autoregressive offline (SenseVoice / Paraformer): single Run.</summary>
    Nar,

    /// <summary>Autoregressive (Whisper via genai); reserved, not in v1 core (spec §8.5).</summary>
    Ar
}
