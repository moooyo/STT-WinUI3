using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.UI.Xaml;
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

    /// <summary>EPs offered to the user: CPU + DirectML always (built into Windows ML), plus any
    /// vendor EP (TensorRT-RTX→CUDA / OpenVINO / QNN / VitisAI) Windows ML registered for this hardware.</summary>
    public ObservableCollection<EpKind> Providers { get; } = new();

    /// <summary>EPs Windows ML actually discovered on this machine (distinct, hardware-class tagged).</summary>
    public string DetectedProviders => "Detected: " + string.Join(", ",
        Stt.Core.Ep.OrtEpEnumerator.Enumerate().Select(d => $"{d.Kind}/{d.Hardware}").Distinct());

    public ObservableCollection<ModelItemViewModel> OfflineModels { get; } = new();
    public ObservableCollection<ModelItemViewModel> StreamingModels { get; } = new();

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(FirstPassVisibility))]
    [NotifyPropertyChangedFor(nameof(SecondPassVisibility))]
    private PipelineMode _mode;
    [ObservableProperty] private EpKind _ep;
    [ObservableProperty] private string? _secondPassModelId;
    [ObservableProperty] private string? _firstPassModelId;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(VadModelDescription))]
    private string? _vadModelPath;

    [ObservableProperty] private float _minTrailingSilenceSeconds;
    [ObservableProperty] private float _maxUtteranceSeconds;
    [ObservableProperty] private string _status = string.Empty;
    [ObservableProperty] private string _driverStatus = "CPU + DirectML built in. Click Download to add vendor EPs (NVIDIA→CUDA).";

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(OfflineModelDescription))]
    private bool _hasOfflineModels;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(StreamingModelDescription))]
    private bool _hasStreamingModels;

    /// <summary>Setting-row description for the offline model picker — flips to a hint when none exist.</summary>
    public string OfflineModelDescription => HasOfflineModels
        ? "Re-decodes each segment for the final text (second pass)."
        : "No offline-capable models installed — import one on the Models page.";

    /// <summary>Setting-row description for the streaming model picker — flips to a hint when none exist.</summary>
    public string StreamingModelDescription => HasStreamingModels
        ? "Produces live partial subtitles in one-pass-streaming / two-pass modes (first pass)."
        : "No streaming-capable models installed — import one on the Models page.";

    /// <summary>Setting-row description showing the currently selected VAD model path.</summary>
    public string VadModelDescription =>
        string.IsNullOrEmpty(VadModelPath) ? "No model selected." : VadModelPath;

    /// <summary>First-pass (streaming) picker is only relevant when streaming is in the pipeline.</summary>
    public Visibility FirstPassVisibility =>
        Mode is PipelineMode.OnePassStreaming or PipelineMode.TwoPass ? Visibility.Visible : Visibility.Collapsed;

    /// <summary>Second-pass (offline) picker is only relevant for one-pass-offline or two-pass.</summary>
    public Visibility SecondPassVisibility =>
        Mode is PipelineMode.OnePassOffline or PipelineMode.TwoPass ? Visibility.Visible : Visibility.Collapsed;

    public SettingsViewModel(IModelRegistry registry, SttOptions options)
    {
        _registry = registry;
        _options = options;

        // Dynamic EP list: CPU + DirectML (always built into Windows ML) + any vendor EP this hardware
        // surfaced (NVIDIA→TensorRT-RTX(CUDA), Intel→OpenVINO, Qualcomm→QNN, AMD→VitisAI). No phantom options.
        RebuildProviders();

        _mode = options.Mode;
        if (!Providers.Contains(options.Ep)) options.Ep = EpKind.Cpu;
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

    private void RebuildProviders()
    {
        // Add only — never Clear: clearing the bound collection forces the ComboBox SelectedItem to
        // null and crashes the TwoWay x:Bind to the non-nullable EpKind. EPs only grow, so additive.
        if (!Providers.Contains(EpKind.Cpu)) Providers.Add(EpKind.Cpu);
        if (!Providers.Contains(EpKind.DirectML)) Providers.Add(EpKind.DirectML);
        foreach (var k in Stt.Core.Ep.OrtEpEnumerator.Enumerate().Select(d => d.Kind))
            if (!Providers.Contains(k)) Providers.Add(k);
    }

    /// <summary>
    /// Download + register the certified vendor EP drivers for this hardware via Windows ML
    /// (NVIDIA TensorRT-RTX → CUDA, Intel OpenVINO, etc.), then refresh the picker. First run may
    /// fetch a few hundred MB; needs Win11 24H2+ and a network. CPU + DirectML work regardless.
    /// </summary>
    [RelayCommand]
    private async Task DownloadDrivers()
    {
        DriverStatus = "Downloading + registering execution-provider drivers…";
        try
        {
            var catalog = Microsoft.Windows.AI.MachineLearning.ExecutionProviderCatalog.GetDefault();
            var names = new List<string>();
            foreach (var ep in catalog.FindAllProviders())   // ALL providers incl. preview NVIDIA TRT-RTX, not just certified
            {
                try { await ep.EnsureReadyAsync(); ep.TryRegister(); names.Add(ep.Name); }
                catch { /* one EP failed to fetch/register — keep the rest */ }
            }
            RebuildProviders();
            OnPropertyChanged(nameof(DetectedProviders));
            DriverStatus = names.Count > 0
                ? "Registered EPs: " + string.Join(", ", names)
                : "No vendor EPs available; CPU + DirectML in use.";
        }
        catch (Exception ex)
        {
            DriverStatus = "Driver download failed (CPU + DirectML still available): " + ex.GetType().Name + " — " + ex.Message;
        }
    }

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
