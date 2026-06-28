#!/usr/bin/env python3
"""Generate golden feature vectors for the C# front-end CI test (spec §7.3).

Produces, under tests/golden/:
  input.wav      a 1 s 16 kHz mono tone (also usable for decode-alignment fixtures)
  featsA.bin     Family A (kaldi-fbank Povey, 80-dim) float32 row-major [T, 80]
  featsA.shape   "T 80"
  featsB.bin     Family B (FunASR fbank + LFR(7,6) + CMVN, 560-dim) float32 [T2, 560]
  featsB.shape   "T2 560"

The C# GoldenFeatureTests compares its KaldiFbankFrontend output element-wise:
  max-abs < 1e-3, mean-abs < 1e-4.

This script is NOT run in CI (it needs lhotse/funasr + the model's CMVN stats). Regenerate the
goldens locally when the front-end parameters change, then commit the .bin/.shape files.

Requires: numpy, soundfile, lhotse (Family A), funasr/kaldi_native_fbank (Family B).
"""
import os
import struct
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "tests", "golden")
os.makedirs(OUT, exist_ok=True)


def write_bin(name, arr):
    arr = np.ascontiguousarray(arr.astype(np.float32))
    arr.tofile(os.path.join(OUT, name + ".bin"))
    with open(os.path.join(OUT, name + ".shape"), "w") as f:
        f.write(" ".join(str(d) for d in arr.shape))


def make_tone(seconds=1.0, sr=16000, freq=440.0):
    t = np.arange(int(seconds * sr)) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def main():
    import soundfile as sf
    audio = make_tone()
    sf.write(os.path.join(OUT, "input.wav"), audio, 16000, subtype="PCM_16")

    # Family A — lhotse kaldi-fbank Povey, 80-dim.
    try:
        from lhotse import Fbank, FbankConfig
        from lhotse.features.kaldi.layers import Wav2LogFilterBank  # noqa: F401
        fb = Fbank(FbankConfig(num_filters=80, snip_edges=False))
        featsA = fb.extract(audio, sampling_rate=16000)
        write_bin("featsA", np.asarray(featsA))
        print("featsA:", featsA.shape)
    except Exception as e:  # pragma: no cover
        print("Family A skipped:", e)

    # Family B — FunASR wav_frontend fbank + LFR(7,6) + CMVN.
    try:
        from funasr.frontends.wav_frontend import WavFrontend
        # cmvn file path must point at the model's am.mvn; placeholder here.
        front = WavFrontend(cmvn_file=os.environ["STT_FUNASR_CMVN"], fs=16000,
                            n_mels=80, lfr_m=7, lfr_n=6)
        import torch
        feats, _ = front(torch.from_numpy(audio).unsqueeze(0),
                         torch.tensor([len(audio)]))
        write_bin("featsB", feats.squeeze(0).numpy())
        print("featsB:", feats.shape)
    except Exception as e:  # pragma: no cover
        print("Family B skipped (set STT_FUNASR_CMVN to am.mvn):", e)


if __name__ == "__main__":
    main()
