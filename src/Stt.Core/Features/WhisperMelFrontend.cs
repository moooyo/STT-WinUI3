using Stt.Abstractions.Features;

namespace Stt.Core.Features;

/// <summary>
/// Whisper / Qwen log-mel front-end (spec §7, Family C). Replicates OpenAI Whisper's feature
/// pipeline: pad/trim the audio to a fixed 30 s window, compute the Slaney-mel power spectrogram
/// via the native knf Whisper computer (Hann window, 25 ms / 10 ms, n_fft=400), then apply
/// <c>log10</c> and Whisper's global dynamic-range normalize
/// (<c>log = max(log, log.max() - 8); (log + 4) / 4</c>). Output is row-major <c>[N_FRAMES, nMels]</c>
/// with <c>nMels</c> = 80 (≤ large-v2) / 128 (large-v3) / model-defined (Qwen); the encoder consumes
/// it transposed to <c>[nMels, N_FRAMES]</c>.
/// </summary>
public sealed class WhisperMelFrontend : IFeatureFrontend, IDisposable
{
    private const int SampleRate = 16000;
    private const int ChunkSamples = 30 * SampleRate; // 480000 — Whisper's fixed 30 s window
    private const int NFrames = 3000;                 // Whisper's fixed mel time axis

    private readonly int _nMels;

    public WhisperMelFrontend(int nMels = 80)
    {
        // 80 = Whisper ≤ large-v2; 128 = large-v3; other values exist for Qwen/Distil exports. The
        // dim is metadata-driven (fail loud on a nonsensical value rather than silently guessing 80).
        if (nMels <= 0 || nMels > 512)
            throw new ArgumentException($"Whisper/Qwen n_mels must be a positive bin count, got {nMels}.", nameof(nMels));
        _nMels = nMels;
    }

    public AsrFeatureFamily Family => AsrFeatureFamily.WhisperLogMel;
    public int FeatureDim => _nMels;

    public float[] Extract(ReadOnlySpan<float> pcm16kMono, out int numFrames)
    {
        if (!KaldiNativeFbankInterop.IsAvailable)
            throw new DllNotFoundException(
                "kaldi_native_fbank_shim native library not found. See docs/native/kaldi-native-fbank.md.");

        // Pad (zeros) or trim to exactly 30 s, like Whisper's pad_or_trim.
        var audio = new float[ChunkSamples];
        int copy = Math.Min(pcm16kMono.Length, ChunkSamples);
        pcm16kMono.Slice(0, copy).CopyTo(audio);

        IntPtr h = KaldiNativeFbankInterop.knf_whisper_create(_nMels, SampleRate);
        if (h == IntPtr.Zero) throw new InvalidOperationException("knf_whisper_create returned null.");

        float[] linear;
        int frames, dim;
        try
        {
            KaldiNativeFbankInterop.knf_accept(h, SampleRate, audio, audio.Length);
            KaldiNativeFbankInterop.knf_finish(h);
            frames = KaldiNativeFbankInterop.knf_num_frames_ready(h);
            dim = KaldiNativeFbankInterop.knf_dim(h);
            linear = new float[frames * dim];
            var frameBuf = new float[dim];
            for (int t = 0; t < frames; t++)
            {
                KaldiNativeFbankInterop.knf_get_frame(h, t, frameBuf);
                Array.Copy(frameBuf, 0, linear, t * dim, dim);
            }
        }
        finally
        {
            KaldiNativeFbankInterop.knf_destroy(h);
        }

        // log10 with the standard 1e-10 floor.
        for (int i = 0; i < linear.Length; i++)
            linear[i] = MathF.Log10(MathF.Max(linear[i], 1e-10f));

        // Trim/pad the time axis to exactly N_FRAMES (Whisper drops the trailing STFT frame).
        var mel = new float[NFrames * dim];
        int keep = Math.Min(frames, NFrames);
        Array.Copy(linear, 0, mel, 0, keep * dim);
        if (keep < NFrames)
        {
            // Pad with the log floor so silent padding doesn't perturb the global max.
            for (int i = keep * dim; i < mel.Length; i++) mel[i] = -10f;
        }

        // Whisper global dynamic-range normalize over the whole [nMels, N_FRAMES] chunk.
        float max = float.NegativeInfinity;
        for (int i = 0; i < mel.Length; i++) if (mel[i] > max) max = mel[i];
        float floor = max - 8f;
        for (int i = 0; i < mel.Length; i++)
            mel[i] = (MathF.Max(mel[i], floor) + 4f) / 4f;

        numFrames = NFrames;
        return mel;
    }

    public void Dispose() { }
}
