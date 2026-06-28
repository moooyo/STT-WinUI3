using Stt.Abstractions.Features;

namespace Stt.Core.Features;

/// <summary>
/// Configuration for <see cref="KaldiFbankFrontend"/> (spec §7.2). Family A (icefall Zipformer)
/// uses 80 bins, povey window, dither 0, snip-edges false, no LFR/CMVN. Family B (FunASR
/// SenseVoice) adds LFR(7,6) + CMVN with per-feature stats read from model metadata.
/// </summary>
public sealed record FbankOptions
{
    public int NumBins { get; init; } = 80;
    public int SampleRate { get; init; } = 16000;
    public float Dither { get; init; } = 0f;
    public bool SnipEdges { get; init; } = false;

    /// <summary>"povey" (A) or "hamming" (B).</summary>
    public string WindowType { get; init; } = "povey";

    public float LowFreq { get; init; } = 20f;

    /// <summary>High mel cutoff Hz; ≤0 is interpreted as Nyquist-relative by the shim.</summary>
    public float HighFreq { get; init; } = 0f;

    public bool UsePower { get; init; } = true;
    public bool UseLogFbank { get; init; } = true;

    /// <summary>
    /// Whether the front-end treats input PCM as already normalized to [-1,1] (true) or scales
    /// it by 32768 (false). Read from model metadata; getting this wrong shifts features by a
    /// constant ≈10.4 (the diagnostic in spec §7.3).
    /// </summary>
    public bool NormalizeSamples { get; init; } = true;

    /// <summary>LFR window/shift as (m, n), or null for Family A (no LFR). Family B = (7, 6).</summary>
    public (int M, int N)? Lfr { get; init; }

    /// <summary>CMVN negative-mean vector (length = NumBins*Lfr.M), or null. Required for Family B.</summary>
    public float[]? CmvnNegMean { get; init; }

    /// <summary>CMVN inverse-stddev vector (length = NumBins*Lfr.M), or null. Required for Family B.</summary>
    public float[]? CmvnInvStddev { get; init; }

    /// <summary>The feature family this configuration corresponds to.</summary>
    public AsrFeatureFamily Family => Lfr is null ? AsrFeatureFamily.KaldiFbankPovey : AsrFeatureFamily.KaldiFbankLfrCmvn;

    /// <summary>Final feature dimension after optional LFR (NumBins, or NumBins*m).</summary>
    public int OutputDim => Lfr is { } l ? NumBins * l.M : NumBins;

    /// <summary>Family A preset: 80-bin povey, no LFR/CMVN.</summary>
    public static FbankOptions FamilyA(int numBins = 80) => new() { NumBins = numBins, WindowType = "povey" };

    /// <summary>Family B preset: 80-bin + LFR(7,6) + CMVN (stats must be supplied from metadata).</summary>
    public static FbankOptions FamilyB(float[] negMean, float[] invStddev, int numBins = 80) => new()
    {
        NumBins = numBins,
        WindowType = "hamming",
        Lfr = (7, 6),
        CmvnNegMean = negMean,
        CmvnInvStddev = invStddev,
    };
}
