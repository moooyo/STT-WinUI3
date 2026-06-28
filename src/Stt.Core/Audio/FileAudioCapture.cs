using Stt.Abstractions.Audio;

namespace Stt.Core.Audio;

/// <summary>
/// An <see cref="IAudioCapture"/> that streams a WAV file as 16 kHz mono frames (spec §4, §15:
/// the headless source that lets Core tests feed a WAV through the full recognition chain with
/// no microphone and no UI). When <c>realTime</c> is true, frames are paced to wall-clock so
/// the pipeline behaves as it would live; in tests it is false for instant playback.
/// </summary>
/// <remarks>
/// Each frame owns a freshly allocated, right-sized array. Frame ownership transfers to the
/// subscriber (the pipeline writes the frame into a channel and consumes it later), so this
/// source must not reuse buffers. The pooled-buffer hot-loop optimization (consumer returns the
/// array) is reserved for the real-time WASAPI capture path.
/// </remarks>
public sealed class FileAudioCapture : IAudioCapture
{
    private readonly float[] _samples;     // 16k mono
    private readonly int _frameSamples;
    private readonly bool _realTime;

    public event Action<AudioFrame>? FrameAvailable;

    public FileAudioCapture(string wavPath, int frameSamples = 512, bool realTime = false)
        : this(LoadMono16k(wavPath), frameSamples, realTime) { }

    public FileAudioCapture(float[] mono16k, int frameSamples, bool realTime)
    {
        if (frameSamples <= 0) throw new ArgumentOutOfRangeException(nameof(frameSamples));
        _samples = mono16k;
        _frameSamples = frameSamples;
        _realTime = realTime;
    }

    private static float[] LoadMono16k(string wavPath)
    {
        var wav = WavIo.ReadPcm(wavPath);
        return Resampler.ToMono16k(wav.Interleaved, wav.SampleRate, wav.Channels);
    }

    public async Task StartAsync(CancellationToken ct)
    {
        double frameSeconds = _frameSamples / (double)Resampler.TargetRate;

        for (int offset = 0; offset < _samples.Length; offset += _frameSamples)
        {
            ct.ThrowIfCancellationRequested();
            int count = Math.Min(_frameSamples, _samples.Length - offset);

            var frame = new float[count];
            Array.Copy(_samples, offset, frame, 0, count);
            FrameAvailable?.Invoke(new AudioFrame(frame, count));

            if (_realTime)
                await Task.Delay(TimeSpan.FromSeconds(frameSeconds), ct).ConfigureAwait(false);
        }
    }

    public Task StopAsync() => Task.CompletedTask;

    public void Dispose() { }
}
