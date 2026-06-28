using Stt.Abstractions.Features;
using Stt.Core.Models;

namespace Stt.Core.Features;

/// <summary>
/// Detects the acoustic feature family from a <see cref="ModelProbe"/> (spec §10.1). The input
/// feature dimension is the arbiter (80 / 128 / 560), with <c>model_type</c> and tensor layout
/// resolving the 80-dim ambiguity (fbank vs whisper-80). Per the iron rule (spec §10.2), an
/// unresolved model returns <see cref="AsrFeatureFamily.Auto"/> — the caller must stop and ask,
/// never silently default to fbank-80.
/// </summary>
public static class FeatureFamilyDetector
{
    public static AsrFeatureFamily Detect(ModelProbe probe)
    {
        string mt = probe.ModelType.ToLowerInvariant();

        // 1. Whisper layout (fixed 3000 time axis) is unambiguous.
        if (probe.Layout == FeatureLayout.MelMid || mt.Contains("whisper") || mt.Contains("dolphin") || mt.Contains("firered"))
            return AsrFeatureFamily.WhisperLogMel;

        // 2. FunASR family by the 560-dim arbiter or model type.
        if (probe.FeatureDim == 560 || mt.Contains("sense_voice") || mt.Contains("sensevoice") || mt.Contains("paraformer"))
            return AsrFeatureFamily.KaldiFbankLfrCmvn;

        // 3. NeMo / GigaAM mel.
        if (mt.Contains("nemo") || mt.Contains("gigaam") || mt.Contains("encdec") || mt.Contains("conformer_nemo"))
            return AsrFeatureFamily.NemoMel;

        // 4. MFCC (TeleSpeech / TDNN), 40-dim.
        if (probe.FeatureDim == 40 && (mt.Contains("telespeech") || mt.Contains("mfcc")))
            return AsrFeatureFamily.Mfcc;

        // 5. Raw audio (T-one): rank-2 input (no feature axis).
        if (probe.Layout == FeatureLayout.Unknown && (mt.Contains("t-one") || mt.Contains("tone_raw")))
            return AsrFeatureFamily.RawAudioSamples;

        // 6. kaldi-fbank Povey (icefall Zipformer / CTC / TDNN / wenet) — the 80/128 fbank case.
        //    whisper-80/128 was already caught by the MelMid layout above, so a time-last 80-dim
        //    input is unambiguously fbank. A time-last 128-dim input must also name an fbank-style
        //    architecture to avoid grabbing an ambiguous NeMo/other 128-dim model.
        bool fbankType = mt.Contains("zipformer") || mt.Contains("ctc") || mt.Contains("tdnn") ||
                         mt.Contains("wenet") || mt.Contains("transducer");
        if (probe.Layout == FeatureLayout.TimeLast)
        {
            if (probe.FeatureDim == 80) return AsrFeatureFamily.KaldiFbankPovey;
            if (probe.FeatureDim == 128 && fbankType) return AsrFeatureFamily.KaldiFbankPovey;
        }

        // Unresolved — refuse to guess (spec §10.2 iron rule).
        return AsrFeatureFamily.Auto;
    }
}
