using System.Runtime.InteropServices;
using Microsoft.UI;
using Microsoft.UI.Windowing;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Stt.App.Views;
using Windows.Graphics;

namespace Stt.App;

/// <summary>Shell window: a NavigationView routing a Frame to the three pages (spec §12). Extends
/// content into a custom title bar and sizes itself to the multi-pane layout.</summary>
public sealed partial class MainWindow : Window
{
    [DllImport("user32.dll")]
    private static extern uint GetDpiForWindow(IntPtr hWnd);

    public MainWindow()
    {
        InitializeComponent();
        Title = "Local STT";

        // Fluent custom title bar: draw our own draggable bar and let Mica show through.
        ExtendsContentIntoTitleBar = true;
        SetTitleBar(AppTitleBar);

        // WinUI 3 has no SizeToContent — size the multi-pane (nav + content) window explicitly,
        // converting DIPs to physical pixels for the monitor's DPI.
        var hwnd = Win32Interop.GetWindowFromWindowId(AppWindow.Id);
        double scale = GetDpiForWindow(hwnd) / 96.0;
        AppWindow.Resize(new SizeInt32((int)(1120 * scale), (int)(760 * scale)));

        ContentFrame.Navigate(typeof(MainPage));
    }

    private void Nav_SelectionChanged(NavigationView sender, NavigationViewSelectionChangedEventArgs args)
    {
        if (args.SelectedItem is NavigationViewItem item)
        {
            switch (item.Tag as string)
            {
                case "main": ContentFrame.Navigate(typeof(MainPage)); sender.Header = "Transcribe"; break;
                case "models": ContentFrame.Navigate(typeof(ModelManagerPage)); sender.Header = "Models"; break;
                case "settings": ContentFrame.Navigate(typeof(SettingsPage)); sender.Header = "Settings"; break;
            }
        }
    }
}
