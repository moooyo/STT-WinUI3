using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Stt.App.ViewModels;
using Windows.Storage.Pickers;

namespace Stt.App.Views;

public sealed partial class ModelManagerPage : Page
{
    public ModelManagerViewModel ViewModel { get; }

    public ModelManagerPage()
    {
        ViewModel = App.GetService<ModelManagerViewModel>();
        InitializeComponent();
    }

    // x:Bind static helpers for the validation-result InfoBar.
    public static bool IsNonEmpty(string? value) => !string.IsNullOrWhiteSpace(value);
    public static InfoBarSeverity SeverityFor(bool isError) =>
        isError ? InfoBarSeverity.Error : InfoBarSeverity.Success;

    private async void Import_Click(object sender, RoutedEventArgs e)
    {
        var picker = new FolderPicker();
        picker.FileTypeFilter.Add("*");
        // Unpackaged: the picker must be associated with the window handle.
        var hwnd = WinRT.Interop.WindowNative.GetWindowHandle(App.MainWindowInstance);
        WinRT.Interop.InitializeWithWindow.Initialize(picker, hwnd);

        var folder = await picker.PickSingleFolderAsync();
        if (folder is not null)
            ViewModel.Import(folder.Path);
    }

    private async void Remove_Click(object sender, RoutedEventArgs e)
    {
        if (sender is not Button { Tag: string id }) return;

        // Removing deletes the model folder from disk — confirm first (destructive action).
        var dialog = new ContentDialog
        {
            Title = "Remove model?",
            Content = $"“{id}” will be deleted from disk. This cannot be undone.",
            PrimaryButtonText = "Remove",
            CloseButtonText = "Cancel",
            DefaultButton = ContentDialogButton.Close,
            XamlRoot = XamlRoot,
        };

        if (await dialog.ShowAsync() == ContentDialogResult.Primary)
            ViewModel.RemoveCommand.Execute(id);
    }
}
