using Stt.Abstractions.Features;
using Stt.Core.Features;
using Stt.Core.Models;

namespace Stt.Core.Tests.Models;

public class FeatureFamilyDetectorTests
{
    private static ModelProbe Probe(string modelType, int featDim, FeatureLayout layout, int? nMels = null) =>
        new() { ModelType = modelType, FeatureDim = featDim, Layout = layout, NMels = nMels };

    [Fact]
    public void SenseVoice_560_Is_KaldiFbankLfrCmvn()
    {
        Assert.Equal(AsrFeatureFamily.KaldiFbankLfrCmvn,
            FeatureFamilyDetector.Detect(Probe("sense_voice_ctc", 560, FeatureLayout.TimeLast)));
    }

    [Fact]
    public void Zipformer_80_TimeLast_Is_KaldiFbankPovey()
    {
        Assert.Equal(AsrFeatureFamily.KaldiFbankPovey,
            FeatureFamilyDetector.Detect(Probe("zipformer2", 80, FeatureLayout.TimeLast)));
    }

    [Fact]
    public void Whisper_LargeV3_128_MelMid_Is_WhisperLogMel()
    {
        Assert.Equal(AsrFeatureFamily.WhisperLogMel,
            FeatureFamilyDetector.Detect(Probe("whisper", 128, FeatureLayout.MelMid)));
    }

    [Fact]
    public void Whisper80_MelMid_Distinguished_From_Fbank80_By_Layout()
    {
        // 80-dim but MelMid layout (fixed 3000 time) → Whisper, not fbank.
        Assert.Equal(AsrFeatureFamily.WhisperLogMel,
            FeatureFamilyDetector.Detect(Probe("", 80, FeatureLayout.MelMid)));
        // 80-dim TimeLast → fbank.
        Assert.Equal(AsrFeatureFamily.KaldiFbankPovey,
            FeatureFamilyDetector.Detect(Probe("", 80, FeatureLayout.TimeLast)));
    }

    [Fact]
    public void Paraformer_Is_KaldiFbankLfrCmvn()
    {
        Assert.Equal(AsrFeatureFamily.KaldiFbankLfrCmvn,
            FeatureFamilyDetector.Detect(Probe("paraformer", 560, FeatureLayout.TimeLast)));
    }

    [Fact]
    public void FireRedASR_Is_Not_Whisper()
    {
        // FireRedASR uses kaldi-fbank-80 + CMVN, NOT Whisper Slaney/30s log-mel — it must not
        // route to Family C. A time-last 80-dim fbank model resolves to KaldiFbankPovey, never Whisper.
        var fam = FeatureFamilyDetector.Detect(Probe("firered_aed", 80, FeatureLayout.TimeLast));
        Assert.NotEqual(AsrFeatureFamily.WhisperLogMel, fam);
        Assert.Equal(AsrFeatureFamily.KaldiFbankPovey, fam);
    }

    [Fact]
    public void Unknown_Refuses_To_Default()
    {
        // Unrecognized type, ambiguous dim, no usable layout → Auto (caller must ask).
        Assert.Equal(AsrFeatureFamily.Auto,
            FeatureFamilyDetector.Detect(Probe("mystery_net", 256, FeatureLayout.Unknown)));
    }
}
