using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Stt.Abstractions.Pipeline;
using Stt.App.Services;

namespace Stt.App.ViewModels;

/// <summary>
/// View model for the transcription window (spec §12): live partial (grey) above the committed
/// finals (black), record start/stop, and a "behind" indicator fed by the pipeline's dropped-frame
/// counter. All engine callbacks are already on the UI thread via the IUiDispatcher.
/// </summary>
public partial class MainViewModel : ObservableObject
{
    private readonly TranscriptionService _service;
    private readonly SttOptions _options;

    public ObservableCollection<TranscriptLine> Lines { get; } = new();

    [ObservableProperty] private string _partialText = string.Empty;
    [ObservableProperty] private bool _isRecording;
    [ObservableProperty] private bool _isBehind;
    [ObservableProperty] private string _status = "Idle";

    public MainViewModel(TranscriptionService service, SttOptions options)
    {
        _service = service;
        _options = options;
        _service.Partial += OnPartial;
        _service.Final += OnFinal;
    }

    [RelayCommand]
    private async Task StartAsync()
    {
        if (IsRecording) return;
        try
        {
            Status = "Starting…";
            await _service.StartAsync(_options);
            IsRecording = true;
            Status = "Listening";
        }
        catch (UnauthorizedAccessException)
        {
            Status = "Microphone access denied. Enable it in Windows Settings.";
        }
        catch (Exception ex)
        {
            Status = ex.Message;
        }
    }

    [RelayCommand]
    private async Task StopAsync()
    {
        if (!IsRecording) return;
        await _service.StopAsync();
        IsRecording = false;
        PartialText = string.Empty;
        Status = "Stopped";
    }

    [RelayCommand]
    private void Clear()
    {
        Lines.Clear();
        PartialText = string.Empty;
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
    }
}
