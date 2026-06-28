using Stt.Abstractions.Features;
using Stt.Abstractions.Models;
using Stt.Core.Features;
using Stt.Core.Text;

namespace Stt.Core.Models;

/// <summary>
/// The five-layer fail-loud load-time validation (spec §10.2). Gates 1–4 are implemented here as
/// pure logic over a manifest + <see cref="ModelProbe"/> + tokens; gate 5 (the optional built-in
/// WAV self-test) runs in <see cref="ModelLoader"/> with a live session. The iron rule: an
/// unresolved family or a missing required parameter is rejected with the checklist, never run.
/// </summary>
public static class ModelValidation
{
    public static ValidationReport Validate(ModelManifest manifest, ModelProbe probe, TokenTable? tokens)
    {
        var report = new ValidationReport();

        // Gate 1: auto-detect the family and confront it with the model's declaration.
        AsrFeatureFamily detected = FeatureFamilyDetector.Detect(probe);
        report.Check($"family auto-detected = {detected} (model_type='{probe.ModelType}', inputDim={probe.FeatureDim}, layout={probe.Layout})");
        if (detected == AsrFeatureFamily.Auto)
            report.Fail("Feature family is UNKNOWN — refusing to default to fbank-80. Provide a manifest 'feature.family'.");

        AsrFeatureFamily declared = ParseFamily(manifest.Feature.Family);
        if (declared != AsrFeatureFamily.Auto && detected != AsrFeatureFamily.Auto && declared != detected)
            report.Fail($"Declared family {declared} contradicts detected {detected} (the graph input is the arbiter).");

        // Gate 2: declared feature dim must equal the encoder's actual input dim.
        if (probe.FeatureDim > 0)
        {
            report.Check($"expected feature dim {manifest.Feature.FeatureDim} vs encoder input dim {probe.FeatureDim}");
            if (manifest.Feature.FeatureDim != probe.FeatureDim)
                report.Fail($"Feature dim mismatch: manifest {manifest.Feature.FeatureDim} != encoder input {probe.FeatureDim} (user may annotate ambiguity but cannot violate the graph).");
        }

        // Gate 3: required parameters per family.
        AsrFeatureFamily fam = detected != AsrFeatureFamily.Auto ? detected : declared;
        if (fam == AsrFeatureFamily.KaldiFbankLfrCmvn)
        {
            int[]? lfr = manifest.Feature.Lfr;
            if (lfr is null || lfr.Length != 2)
                report.Fail("Family B (LFR+CMVN) requires feature.lfr = [m, n] (e.g. [7,6]).");
            else
            {
                int baseBins = manifest.Feature.FeatureDim / lfr[0];
                report.Check($"LFR geometry: {baseBins} bins × {lfr[0]} = {manifest.Feature.FeatureDim}");
                if (baseBins * lfr[0] != manifest.Feature.FeatureDim)
                    report.Fail($"LFR geometry invalid: baseBins({baseBins}) × m({lfr[0]}) != featDim({manifest.Feature.FeatureDim}).");
            }
            if (string.Equals(manifest.Feature.Cmvn, "none", StringComparison.OrdinalIgnoreCase) && !manifest.Capabilities.NeedsLfrCmvn)
                report.Fail("Family B requires CMVN stats (feature.cmvn must not be 'none').");
        }
        else if (fam == AsrFeatureFamily.WhisperLogMel)
        {
            if (probe.NMels is { } n && manifest.Feature.FeatureDim != n)
                report.Fail($"Whisper n_mels mismatch: manifest {manifest.Feature.FeatureDim} != metadata {n}.");
        }

        // Gate 4: tokens.txt line count must equal vocab_size.
        if (tokens is not null && probe.VocabSize > 0)
        {
            report.Check($"tokens count {tokens.Count} vs vocab_size {probe.VocabSize}");
            if (tokens.Count != probe.VocabSize)
                report.Fail($"tokens.txt has {tokens.Count} entries but vocab_size is {probe.VocabSize} (wrong tokens file).");
        }

        return report;
    }

    private static AsrFeatureFamily ParseFamily(string s) =>
        Enum.TryParse<AsrFeatureFamily>(s, ignoreCase: true, out var f) ? f : AsrFeatureFamily.Auto;
}
