using NAudio.CoreAudioApi;
using NAudio.Wave;
using Stt.Abstractions.Audio;
using Stt.Core.Audio;

namespace Stt.Audio.Windows;

/// <summary>A capture endpoint for the UI device picker. An empty <see cref="Id"/> means "system default".</summary>
public readonly record struct AudioCaptureDeviceInfo(string Id, string Name);

/// <summary>
/// WASAPI microphone capture (spec §4: the only OS-bound audio dependency, isolated here). Opens
/// the default — or a selected — capture endpoint via NAudio, converts the device mix format to
/// 16 kHz mono float, and raises <see cref="IAudioCapture.FrameAvailable"/> in fixed-size frames.
/// Microphone-denied surfaces as <see cref="UnauthorizedAccessException"/> from
/// <see cref="StartAsync"/> (spec §12, §14: the app prompts and stops recording).
/// </summary>
public sealed class WasapiAudioCapture : IAudioCapture
{
    private readonly MMDevice? _device;
    private readonly int _frameSamples;
    private WasapiCapture? _capture;
    private FrameChunker? _chunker;
    private TaskCompletionSource<bool>? _stopped;
    private int _srcRate;
    private int _srcChannels;
    private bool _isFloat;
    private int _bytesPerSample;

    public event Action<AudioFrame>? FrameAvailable;

    /// <param name="device">Capture endpoint, or null for the system default.</param>
    /// <param name="frameSamples">Samples per emitted 16 kHz mono frame (default 512 = one VAD window).</param>
    public WasapiAudioCapture(MMDevice? device = null, int frameSamples = 512)
    {
        _device = device;
        _frameSamples = frameSamples;
    }

    /// <summary>Enumerate active capture (microphone) endpoints for the UI device picker.</summary>
    public static IReadOnlyList<MMDevice> EnumerateCaptureDevices()
    {
        using var enumerator = new MMDeviceEnumerator();
        return enumerator.EnumerateAudioEndPoints(DataFlow.Capture, DeviceState.Active).ToList();
    }

    /// <summary>
    /// Enumerate capture endpoints as lightweight value records (id + friendly name) so the UI can
    /// bind a picker without holding live COM <see cref="MMDevice"/> handles. Returns an empty list
    /// if the audio subsystem is unavailable.
    /// </summary>
    public static IReadOnlyList<AudioCaptureDeviceInfo> EnumerateCaptureDeviceInfos()
    {
        var list = new List<AudioCaptureDeviceInfo>();
        try
        {
            using var enumerator = new MMDeviceEnumerator();
            foreach (var device in enumerator.EnumerateAudioEndPoints(DataFlow.Capture, DeviceState.Active))
                using (device)
                    list.Add(new AudioCaptureDeviceInfo(device.ID, device.FriendlyName));
        }
        catch { /* no audio subsystem / access denied → no selectable devices */ }
        return list;
    }

    /// <summary>Resolve a capture endpoint by id, or null for the system default / unknown id.</summary>
    public static MMDevice? GetDeviceById(string? id)
    {
        if (string.IsNullOrEmpty(id)) return null;
        try
        {
            using var enumerator = new MMDeviceEnumerator();
            return enumerator.GetDevice(id);
        }
        catch { return null; }
    }

    public Task StartAsync(CancellationToken ct)
    {
        _capture = _device is null ? new WasapiCapture() : new WasapiCapture(_device);
        WaveFormat fmt = _capture.WaveFormat;
        _srcRate = fmt.SampleRate;
        _srcChannels = fmt.Channels;
        _isFloat = fmt.Encoding == WaveFormatEncoding.IeeeFloat;
        _bytesPerSample = fmt.BitsPerSample / 8;
        _chunker = new FrameChunker(_frameSamples, _frameSamples);
        _stopped = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);

        _capture.DataAvailable += OnData;
        _capture.RecordingStopped += OnStopped;

        ct.Register(() => { try { _capture?.StopRecording(); } catch { /* already stopping */ } });

        try
        {
            _capture.StartRecording();
        }
        catch (Exception ex) when (ex is UnauthorizedAccessException || IsAccessDenied(ex))
        {
            throw new UnauthorizedAccessException("Microphone access was denied.", ex);
        }

        return _stopped.Task;
    }

    private static bool IsAccessDenied(Exception ex) =>
        ex.HResult == unchecked((int)0x80070005); // E_ACCESSDENIED

    private void OnData(object? sender, WaveInEventArgs e)
    {
        if (_chunker is null) return;

        float[] interleaved = DecodeToFloat(e.Buffer, e.BytesRecorded);
        float[] mono16k = Resampler.ToMono16k(interleaved, _srcRate, _srcChannels);

        _chunker.Push(mono16k);
        while (_chunker.TryGetWindow(out float[] window))
            FrameAvailable?.Invoke(new AudioFrame(window, window.Length));
    }

    private float[] DecodeToFloat(byte[] buffer, int bytesRecorded)
    {
        int count = bytesRecorded / _bytesPerSample;
        var samples = new float[count];
        if (_isFloat && _bytesPerSample == 4)
        {
            Buffer.BlockCopy(buffer, 0, samples, 0, count * 4);
        }
        else if (_bytesPerSample == 2) // 16-bit PCM
        {
            for (int i = 0; i < count; i++)
            {
                short s = (short)(buffer[i * 2] | (buffer[i * 2 + 1] << 8));
                samples[i] = s / 32768f;
            }
        }
        else if (_bytesPerSample == 4) // 32-bit PCM int
        {
            for (int i = 0; i < count; i++)
            {
                int s = BitConverter.ToInt32(buffer, i * 4);
                samples[i] = s / 2147483648f;
            }
        }
        return samples;
    }

    private void OnStopped(object? sender, StoppedEventArgs e)
    {
        if (e.Exception is not null) _stopped?.TrySetException(e.Exception);
        else _stopped?.TrySetResult(true);
    }

    public Task StopAsync()
    {
        try { _capture?.StopRecording(); } catch { /* already stopped */ }
        return _stopped?.Task ?? Task.CompletedTask;
    }

    public void Dispose()
    {
        if (_capture is not null)
        {
            _capture.DataAvailable -= OnData;
            _capture.RecordingStopped -= OnStopped;
            _capture.Dispose();
            _capture = null;
        }
        _device?.Dispose();
    }
}
