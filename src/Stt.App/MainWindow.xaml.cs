using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Stt.App.Views;

namespace Stt.App;

/// <summary>Shell window: a NavigationView routing a Frame to the three pages (spec §12).</summary>
public sealed partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();
        Title = "Local STT";
        ContentFrame.Navigate(typeof(MainPage));
    }

    private void Nav_SelectionChanged(NavigationView sender, NavigationViewSelectionChangedEventArgs args)
    {
        if (args.SelectedItem is NavigationViewItem item)
        {
            switch (item.Tag as string)
            {
                case "main": ContentFrame.Navigate(typeof(MainPage)); break;
                case "models": ContentFrame.Navigate(typeof(ModelManagerPage)); break;
                case "settings": ContentFrame.Navigate(typeof(SettingsPage)); break;
            }
        }
    }
}
