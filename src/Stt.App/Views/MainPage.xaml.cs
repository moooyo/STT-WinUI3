using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Stt.App.ViewModels;
using Windows.ApplicationModel.DataTransfer;
using Windows.Storage;
using Windows.Storage.Pickers;

namespace Stt.App.Views;

public sealed partial class MainPage : Page
{
    public MainViewModel ViewModel { get; }

    public MainPage()
    {
        ViewModel = App.GetService<MainViewModel>();
        InitializeComponent();
        ViewModel.FinalCommitted += ScrollToLatest;
    }

    // x:Bind static helpers (preferred over IValueConverter for simple cases — skill guidance).
    public static bool Not(bool value) => !value;
    public static bool IsNonEmpty(string? value) => !string.IsNullOrWhiteSpace(value);

    private void ScrollToLatest()
    {
        if (ViewModel.Lines.Count > 0)
            TranscriptList.ScrollIntoView(ViewModel.Lines[^1]);
    }

    private void Copy_Click(object sender, RoutedEventArgs e)
    {
        var package = new DataPackage();
        package.SetText(ViewModel.ExportText());
        Clipboard.SetContent(package);
        ViewModel.Status = "Copied to clipboard.";
    }

    private async void Export_Click(object sender, RoutedEventArgs e)
    {
        var picker = new FileSavePicker { SuggestedStartLocation = PickerLocationId.DocumentsLibrary };
        picker.FileTypeChoices.Add("Plain text", new List<string> { ".txt" });
        picker.SuggestedFileName = "transcript";

        // Unpackaged: the picker must be associated with the window handle.
        var hwnd = WinRT.Interop.WindowNative.GetWindowHandle(App.MainWindowInstance);
        WinRT.Interop.InitializeWithWindow.Initialize(picker, hwnd);

        StorageFile? file = await picker.PickSaveFileAsync();
        if (file is not null)
        {
            await FileIO.WriteTextAsync(file, ViewModel.ExportText());
            ViewModel.Status = $"Exported to {file.Name}.";
        }
    }
}
