namespace Stt.Abstractions.Audio;

/// <summary>
/// A source of 16 kHz mono float audio frames (spec §5.1). Implementations: WASAPI capture
/// (app) and <c>FileAudioCapture</c> (headless tests). The <see cref="FrameAvailable"/> event
/// is raised on the producer's thread; handlers must be non-blocking and non-allocating
/// (spec §6: the audio callback thread never calls ORT and never allocates).
/// </summary>
public interface IAudioCapture : IDisposable
{
    /// <summary>Raised per captured frame (16 kHz mono float).</summary>
    event Action<AudioFrame> FrameAvailable;

    /// <summary>Begin capture. Completes when the source ends (file) or is stopped (mic).</summary>
    Task StartAsync(CancellationToken ct);

    /// <summary>Stop capture and release the device.</summary>
    Task StopAsync();
}
