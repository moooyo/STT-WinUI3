using CommunityToolkit.Mvvm.ComponentModel;

namespace Stt.App.ViewModels;

/// <summary>A line in the transcript list. Finals render black; the live partial renders grey/italic.</summary>
public partial class TranscriptLine : ObservableObject
{
    [ObservableProperty] private string _text = string.Empty;
    [ObservableProperty] private bool _isFinal;

    public int SegmentId { get; init; }
}
