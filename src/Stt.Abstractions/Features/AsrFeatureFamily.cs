namespace Stt.Abstractions.Features;

/// <summary>
/// Acoustic feature family (spec §7.1). The family dictates which front-end extracts features
/// and with what parameters. Picking the wrong family produces silent garbage (the encoder
/// still emits plausible-but-wrong logits), so the family is detected from model metadata and
/// hard-validated at load time (spec §10.2).
/// </summary>
public enum AsrFeatureFamily
{
    /// <summary>Unknown — must be resolved before use; never silently treated as fbank-80.</summary>
    Auto,

    /// <summary>A: kaldi-fbank Povey/HTK mel, natural log, no CMVN. 80-dim. icefall Zipformer/CTC.</summary>
    KaldiFbankPovey,

    /// <summary>B: kaldi-fbank + LFR + CMVN. 560 = 80×7. FunASR Paraformer / SenseVoice.</summary>
    KaldiFbankLfrCmvn,

    /// <summary>C: Whisper log-mel (Slaney/Hann/log10), fixed 30 s. 80 or 128. Whisper/Dolphin.</summary>
    WhisperLogMel,

    /// <summary>D: NeMo mel (kaldi-fbank Slaney+Hann) + per-feature normalization. NeMo/GigaAM.</summary>
    NemoMel,

    /// <summary>E: MFCC, num_ceps=40. TeleSpeech/TDNN.</summary>
    Mfcc,

    /// <summary>F: raw PCM, no fbank. T-one.</summary>
    RawAudioSamples
}
