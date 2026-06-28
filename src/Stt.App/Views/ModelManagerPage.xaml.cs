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

    private void Remove_Click(object sender, RoutedEventArgs e)
    {
        if (sender is Button { Tag: string id })
            ViewModel.RemoveCommand.Execute(id);
    }
}
