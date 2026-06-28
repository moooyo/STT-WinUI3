using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Stt.App.ViewModels;
using Windows.Storage.Pickers;

namespace Stt.App.Views;

public sealed partial class SettingsPage : Page
{
    public SettingsViewModel ViewModel { get; }

    public SettingsPage()
    {
        ViewModel = App.GetService<SettingsViewModel>();
        InitializeComponent();
    }

    // x:Bind static helper for driving the save-confirmation InfoBar.
    public static bool IsNonEmpty(string? value) => !string.IsNullOrWhiteSpace(value);

    private async void BrowseVad_Click(object sender, RoutedEventArgs e)
    {
        var picker = new FileOpenPicker();
        picker.FileTypeFilter.Add(".onnx");
        var hwnd = WinRT.Interop.WindowNative.GetWindowHandle(App.MainWindowInstance);
        WinRT.Interop.InitializeWithWindow.Initialize(picker, hwnd);

        var file = await picker.PickSingleFileAsync();
        if (file is not null)
            ViewModel.SetVadModelPath(file.Path);
    }
}
