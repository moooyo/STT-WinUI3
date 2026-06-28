# Golden feature vectors

This directory holds reference acoustic features used by `GoldenFeatureTests` to verify the C#
front-ends bit-match the Python authorities (spec §7.3): **max-abs < 1e-3, mean-abs < 1e-4**.

## Files (generated, committed when present)

| File | Meaning |
|---|---|
| `input.wav` | 1 s 16 kHz mono tone fed to both front-ends |
| `featsA.bin` / `featsA.shape` | Family A: kaldi-fbank Povey 80-dim `[T, 80]` float32 row-major |
| `featsB.bin` / `featsB.shape` | Family B: FunASR fbank + LFR(7,6) + CMVN 560-dim `[T2, 560]` float32 |

`.bin` files are little-endian float32, row-major; `.shape` is space-separated dimensions.

> Note: `*.bin` is git-ignored by default (model-artifact rule). Force-add the goldens when you
> want them in CI: `git add -f tests/golden/featsA.bin tests/golden/featsA.shape ...`.

## Regenerating

```bash
pip install numpy soundfile lhotse funasr torch
# Family B needs the model's CMVN stats (am.mvn):
export STT_FUNASR_CMVN=/path/to/sense-voice/am.mvn
python scripts/gen_golden_features.py
```

Then run the test (it auto-detects the goldens + the native fbank shim):

```bash
dotnet test --filter GoldenFeature
```

## Diagnostics (spec §7.3)

- A constant offset ≈ 10.4 between C# and Python → the amplitude (×32768) `normalize_samples`
  switch is wrong.
- Error that scales with amplitude → mel/power configuration mismatch.
- Error only at frame edges → `snip_edges` / padding mismatch.
