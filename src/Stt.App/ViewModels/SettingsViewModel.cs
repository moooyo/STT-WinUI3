using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Stt.Abstractions.Ep;
using Stt.Abstractions.Models;
using Stt.Abstractions.Pipeline;
using Stt.App.Services;

namespace Stt.App.ViewModels;

/// <summary>
/// View model for the pipeline + settings page (spec §12): mode, per-pass model assignment (illegal
/// combos are filtered by capability flags), EP preference, and VAD/endpoint thresholds. Persists to
/// <see cref="SttOptions"/>.
/// </summary>
public partial class SettingsViewModel : ObservableObject
{
    private readonly IModelRegistry _registry;
    private readonly SttOptions _options;

    public IReadOnlyList<PipelineMode> Modes { get; } =
        new[] { PipelineMode.OnePassOffline, PipelineMode.OnePassStreaming, PipelineMode.TwoPass };

    public IReadOnlyList<EpKind> Providers { get; } =
        new[] { EpKind.Cpu, EpKind.DirectML };

    public ObservableCollection<ModelItemViewModel> OfflineModels { get; } = new();
    public ObservableCollection<ModelItemViewModel> StreamingModels { get; } = new();

    [ObservableProperty] private PipelineMode _mode;
    [ObservableProperty] private EpKind _ep;
    [ObservableProperty] private string? _secondPassModelId;
    [ObservableProperty] private string? _firstPassModelId;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(VadModelDescription))]
    private string? _vadModelPath;

    [ObservableProperty] private float _minTrailingSilenceSeconds;
    [ObservableProperty] private float _maxUtteranceSeconds;
    [ObservableProperty] private string _status = string.Empty;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(OfflineModelDescription))]
    private bool _hasOfflineModels;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(StreamingModelDescription))]
    private bool _hasStreamingModels;

    /// <summary>SettingsCard description for the offline model picker — flips to a hint when none exist.</summary>
    public string OfflineModelDescription => HasOfflineModels
        ? "Re-decodes each segment for the final text (second pass)."
        : "No offline-capable models installed — import one on the Models page.";

    /// <summary>SettingsCard description for the streaming model picker — flips to a hint when none exist.</summary>
    public string StreamingModelDescription => HasStreamingModels
        ? "Produces live partial subtitles in one-pass-streaming / two-pass modes (first pass)."
        : "No streaming-capable models installed — import one on the Models page.";

    /// <summary>SettingsCard description showing the currently selected VAD model path.</summary>
    public string VadModelDescription =>
        string.IsNullOrEmpty(VadModelPath) ? "No model selected." : VadModelPath;

    public SettingsViewModel(IModelRegistry registry, SttOptions options)
    {
        _registry = registry;
        _options = options;

        _mode = options.Mode;
        _ep = options.Ep;
        _secondPassModelId = options.SecondPassModelId;
        _firstPassModelId = options.FirstPassModelId;
        _vadModelPath = options.VadModelPath;
        _minTrailingSilenceSeconds = options.MinTrailingSilenceSeconds;
        _maxUtteranceSeconds = options.MaxUtteranceSeconds;

        RefreshModelLists();
    }

    public void RefreshModelLists()
    {
        OfflineModels.Clear();
        StreamingModels.Clear();
        foreach (var m in _registry.List())
        {
            var vm = ModelItemViewModel.From(m);
            if (vm.Offline) OfflineModels.Add(vm);
            if (vm.Streaming) StreamingModels.Add(vm);
        }
        HasOfflineModels = OfflineModels.Count > 0;
        HasStreamingModels = StreamingModels.Count > 0;
    }

    /// <summary>Set the VAD model path (called from the page after a FileOpenPicker).</summary>
    public void SetVadModelPath(string path) => VadModelPath = path;

    [RelayCommand]
    private void Save()
    {
        _options.Mode = Mode;
        _options.Ep = Ep;
        _options.SecondPassModelId = SecondPassModelId;
        _options.FirstPassModelId = FirstPassModelId;
        _options.VadModelPath = VadModelPath;
        _options.MinTrailingSilenceSeconds = MinTrailingSilenceSeconds;
        _options.MaxUtteranceSeconds = MaxUtteranceSeconds;
        _options.Save();
        Status = "Settings saved.";
    }
}
