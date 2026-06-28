using Microsoft.UI.Xaml.Controls;
using Stt.App.ViewModels;

namespace Stt.App.Views;

public sealed partial class MainPage : Page
{
    public MainViewModel ViewModel { get; }

    public MainPage()
    {
        ViewModel = App.GetService<MainViewModel>();
        InitializeComponent();
    }
}
