using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Stt.Abstractions.Pipeline;
using Stt.App.Services;
using Stt.Audio.Windows;

namespace Stt.App.ViewModels;

/// <summary>
/// View model for the transcription window (spec §12): live partial (grey) above the committed
/// finals (black), record start/stop, microphone selection, copy/export, and a "behind" indicator
/// fed by the pipeline's dropped-frame counter. All engine callbacks are already on the UI thread
/// via the IUiDispatcher. Command enablement is driven by <see cref="IsRecording"/> and
/// <see cref="HasContent"/> so buttons disable themselves (no IsEnabled converters).
/// </summary>
public partial class MainViewModel : ObservableObject
{
    private readonly TranscriptionService _service;
    private readonly SttOptions _options;

    public ObservableCollection<TranscriptLine> Lines { get; } = new();

    /// <summary>Active capture endpoints; the first entry is the system default (empty id).</summary>
    public ObservableCollection<AudioCaptureDeviceInfo> Devices { get; } = new();

    [ObservableProperty] private string _partialText = string.Empty;

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(StartCommand))]
    [NotifyCanExecuteChangedFor(nameof(StopCommand))]
    private bool _isRecording;

    [ObservableProperty] private bool _isBehind;
    [ObservableProperty] private string _status = "Idle";

    /// <summary>Non-empty only for actionable failures (mic denied, missing model) — shown in an InfoBar.</summary>
    [ObservableProperty] private string _errorMessage = string.Empty;

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(ClearCommand))]
    private bool _hasContent;

    [ObservableProperty] private string? _selectedDeviceId;

    public MainViewModel(TranscriptionService service, SttOptions options)
    {
        _service = service;
        _options = options;
        _service.Partial += OnPartial;
        _service.Final += OnFinal;
        LoadDevices();
    }

    /// <summary>Raised after a final line is committed so the view can auto-scroll to the newest line.</summary>
    public event Action? FinalCommitted;

    private void LoadDevices()
    {
        Devices.Add(new AudioCaptureDeviceInfo(string.Empty, "System default microphone"));
        foreach (var d in WasapiAudioCapture.EnumerateCaptureDeviceInfos())
            Devices.Add(d);
        SelectedDeviceId = string.Empty;
    }

    private bool CanStart() => !IsRecording;
    private bool CanStop() => IsRecording;
    private bool CanModify() => HasContent;

    [RelayCommand(CanExecute = nameof(CanStart))]
    private async Task StartAsync()
    {
        ErrorMessage = string.Empty;
        try
        {
            Status = "Starting…";
            await _service.StartAsync(_options, SelectedDeviceId);
            IsRecording = true;
            Status = "Listening";
        }
        catch (UnauthorizedAccessException)
        {
            Status = "Idle";
            ErrorMessage = "Microphone access was denied. Enable microphone access for this app in " +
                           "Windows Settings → Privacy & security → Microphone, then try again.";
        }
        catch (Exception ex)
        {
            Status = "Idle";
            ErrorMessage = ex.Message;
        }
    }

    [RelayCommand(CanExecute = nameof(CanStop))]
    private async Task StopAsync()
    {
        await _service.StopAsync();
        IsRecording = false;
        PartialText = string.Empty;
        Status = "Stopped";
    }

    [RelayCommand(CanExecute = nameof(CanModify))]
    private void Clear()
    {
        Lines.Clear();
        PartialText = string.Empty;
        HasContent = false;
    }

    public string ExportText() => string.Join(Environment.NewLine, Lines.Select(l => l.Text));

    private void OnPartial(PartialResult p)
    {
        PartialText = p.Text;
        IsBehind = _service.DroppedFrames > 0;
    }

    private void OnFinal(FinalResult f)
    {
        // Replace an existing partial for this segment, or append a new final line.
        var existing = Lines.FirstOrDefault(l => l.SegmentId == f.SegmentId);
        if (existing is not null)
        {
            existing.Text = f.Text;
            existing.IsFinal = true;
        }
        else if (!string.IsNullOrWhiteSpace(f.Text))
        {
            Lines.Add(new TranscriptLine { SegmentId = f.SegmentId, Text = f.Text, IsFinal = true });
        }
        PartialText = string.Empty;
        IsBehind = _service.DroppedFrames > 0;
        HasContent = Lines.Count > 0;
        FinalCommitted?.Invoke();
    }
}
