using System.Runtime.InteropServices;

namespace Stt.Core.Features;

/// <summary>
/// P/Invoke bindings for the <c>kaldi-native-fbank</c> online fbank extractor (spec §7.2, D6).
/// </summary>
/// <remarks>
/// The project owns a thin C ABI shim (<c>kaldi_native_fbank_shim</c>) that wraps knf's C++
/// <c>knf::OnlineFbank</c>; see <c>docs/native/kaldi-native-fbank.md</c> for the shim source and
/// build instructions. The shim is bit-compatible with the training-time fbank (Apache-2.0,
/// win-x64 / win-arm64). When the native library is absent the bindings throw
/// <see cref="DllNotFoundException"/>; callers probe <see cref="IsAvailable"/> first.
/// </remarks>
internal static partial class KaldiNativeFbankInterop
{
    private const string Lib = "kaldi_native_fbank_shim";

    /// <summary>
    /// Create an online fbank extractor.
    /// </summary>
    /// <param name="numBins">Mel bins (feature dim per frame), e.g. 80.</param>
    /// <param name="sampleRate">Expected sample rate, e.g. 16000.</param>
    /// <param name="dither">Dither amount; 0 for deterministic ASR features.</param>
    /// <param name="snipEdges">1 = snip edges (fewer frames), 0 = pad (kaldi-fbank ASR default 0).</param>
    /// <param name="windowType">"povey" (A) or "hamming" (B). Marshalled as UTF-8.</param>
    /// <param name="lowFreq">Mel low cutoff Hz (e.g. 20).</param>
    /// <param name="highFreq">Mel high cutoff Hz; ≤0 means Nyquist-relative (e.g. -400 → sr/2-400).</param>
    /// <param name="usePower">1 = power spectrum, 0 = magnitude.</param>
    /// <param name="useLog">1 = natural-log fbank (ASR default).</param>
    /// <param name="normalizeSamples">1 = treat input as already in [-1,1]; 0 = scale by 32768.</param>
    [LibraryImport(Lib, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial IntPtr knf_create(
        int numBins, float sampleRate, float dither, int snipEdges, string windowType,
        float lowFreq, float highFreq, int usePower, int useLog, int normalizeSamples);

    /// <summary>
    /// Create a librosa/Slaney-mel fbank extractor (Family D, NeMo/GigaAM). Same handle ABI as
    /// <see cref="knf_create"/>; the mel filterbank uses librosa's Slaney scale + norm.
    /// </summary>
    [LibraryImport(Lib, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial IntPtr knf_mel_create(
        int numBins, float sampleRate, float dither, int snipEdges, string windowType,
        float lowFreq, float highFreq, int usePower, int useLog);

    /// <summary>
    /// Create a Whisper log-mel extractor (Family C). <paramref name="dim"/> = 80 (≤ large-v2) or
    /// 128 (large-v3). Output is LINEAR mel energy per frame; the caller applies log10 + Whisper's
    /// global dynamic-range normalize. Same handle ABI as <see cref="knf_create"/>.
    /// </summary>
    [LibraryImport(Lib)]
    internal static partial IntPtr knf_whisper_create(int dim, float sampleRate);

    [LibraryImport(Lib)]
    internal static partial void knf_accept(IntPtr handle, float sampleRate, [In] float[] samples, int count);

    [LibraryImport(Lib)]
    internal static partial void knf_finish(IntPtr handle);

    [LibraryImport(Lib)]
    internal static partial int knf_num_frames_ready(IntPtr handle);

    [LibraryImport(Lib)]
    internal static partial int knf_dim(IntPtr handle);

    /// <summary>Copy frame <paramref name="frameIndex"/> (length = <c>knf_dim</c>) into <paramref name="outFrame"/>.</summary>
    [LibraryImport(Lib)]
    internal static partial void knf_get_frame(IntPtr handle, int frameIndex, [Out] float[] outFrame);

    [LibraryImport(Lib)]
    internal static partial void knf_destroy(IntPtr handle);

    private static bool? _available;

    /// <summary>True when the native shim can be loaded. Cached after first probe.</summary>
    internal static bool IsAvailable
    {
        get
        {
            if (_available.HasValue) return _available.Value;
            _available = NativeLibrary.TryLoad(Lib, typeof(KaldiNativeFbankInterop).Assembly, null, out _);
            return _available.Value;
        }
    }
}
