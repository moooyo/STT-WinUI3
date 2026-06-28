using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Stt.Abstractions.Models;

namespace Stt.App.ViewModels;

/// <summary>A model row in the Model Manager with capability badges (spec §12).</summary>
public sealed class ModelItemViewModel
{
    public required string Id { get; init; }
    public required string DisplayName { get; init; }
    public required string Family { get; init; }

    /// <summary>Feature family letter (A/B/C/D) — the acoustic front-end the engine will use.</summary>
    public required string FeatureFamily { get; init; }

    /// <summary>Friendly model kind shown in the list, e.g. "Whisper · family C".</summary>
    public required string Kind { get; init; }

    public bool Streaming { get; init; }
    public bool Offline { get; init; }
    public bool Multilingual { get; init; }

    public string Badges
    {
        get
        {
            var b = new List<string>();
            if (Streaming) b.Add("streaming");
            if (Offline) b.Add("offline");
            if (Multilingual) b.Add("multilingual");
            return string.Join(" · ", b);
        }
    }

    /// <summary>Capability badges as discrete chips for the Model Manager list (feature family first).</summary>
    public IReadOnlyList<string> BadgeList
    {
        get
        {
            var b = new List<string>();
            if (!string.IsNullOrEmpty(FeatureFamily)) b.Add(FeatureFamily);
            if (Streaming) b.Add("streaming");
            if (Offline) b.Add("offline");
            if (Multilingual) b.Add("multilingual");
            return b;
        }
    }

    public static ModelItemViewModel From(ModelManifest m) => new()
    {
        Id = m.Id,
        DisplayName = string.IsNullOrEmpty(m.DisplayName) ? m.Id : m.DisplayName,
        Family = m.Family,
        FeatureFamily = FeatureFamilyLetter(m.Feature.Family),
        Kind = FriendlyKind(m.Family, m.Feature.Family),
        Streaming = m.Capabilities.StreamingCapable,
        Offline = m.Capabilities.OfflineCapable,
        Multilingual = m.Capabilities.Multilingual,
    };

    private static string FeatureFamilyLetter(string featFamily) => featFamily switch
    {
        "KaldiFbankPovey" => "A",
        "KaldiFbankLfrCmvn" => "B",
        "WhisperLogMel" => "C",
        "NemoMel" => "D",
        _ => "",
    };

    private static string FriendlyKind(string family, string featFamily)
    {
        string fam = family.ToLowerInvariant() switch
        {
            "whisper" => "Whisper",
            "nemo" => "NeMo",
            "sense_voice" => "SenseVoice",
            "paraformer" => "Paraformer",
            "transducer" => "Zipformer/transducer",
            "ctc" => "CTC",
            _ => string.IsNullOrEmpty(family) ? "model" : family,
        };
        string letter = FeatureFamilyLetter(featFamily);
        return letter.Length > 0 ? $"{fam} · family {letter}" : fam;
    }
}

/// <summary>
/// View model for the Model Manager page (spec §12): installed list, sideload import (folder),
/// capability badges, and removal. The folder picker is invoked from code-behind (needs the
/// window handle in an unpackaged app) and the path is handed to <see cref="Import"/>.
/// </summary>
public partial class ModelManagerViewModel : ObservableObject
{
    private readonly IModelRegistry _registry;

    public ObservableCollection<ModelItemViewModel> Models { get; } = new();

    [ObservableProperty] private string _status = string.Empty;
    [ObservableProperty] private bool _isError;
    [ObservableProperty] private bool _hasModels;

    public ModelManagerViewModel(IModelRegistry registry)
    {
        _registry = registry;
        Refresh();
    }

    public void Refresh()
    {
        Models.Clear();
        foreach (var m in _registry.List())
            Models.Add(ModelItemViewModel.From(m));
        HasModels = Models.Count > 0;
    }

    /// <summary>Import a sideloaded model folder (called from the page after a FolderPicker).</summary>
    public void Import(string folderPath)
    {
        try
        {
            var m = _registry.ImportFromFolder(folderPath);
            IsError = false;
            Status = $"Imported '{m.DisplayName}' ({m.Family}). Validation passed.";
            Refresh();
        }
        catch (Exception ex)
        {
            IsError = true;
            Status = $"Import failed: {ex.Message}";
        }
    }

    [RelayCommand]
    private void Remove(string id)
    {
        _registry.Remove(id);
        Refresh();
    }
}
