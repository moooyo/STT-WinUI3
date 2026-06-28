using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;

namespace Stt.App.Controls;

/// <summary>
/// A lightweight Fluent "settings row": a <see cref="Header"/> + <see cref="Description"/> on the
/// left and a single control (the <see cref="ContentControl.Content"/>) on the right, wrapped in a
/// card surface. This is a native replacement for CommunityToolkit's <c>SettingsCard</c> so the app
/// depends only on the Windows App SDK — the toolkit's WinUI controls trail the SDK by a major
/// version (no Windows App SDK 2.x build exists), and we are on the latest SDK. The default template
/// lives in <c>Themes/Generic.xaml</c>.
/// </summary>
public sealed class SettingCard : ContentControl
{
    public SettingCard()
    {
        DefaultStyleKey = typeof(SettingCard);
    }

    /// <summary>Primary label shown in the body type ramp.</summary>
    public string Header
    {
        get => (string)GetValue(HeaderProperty);
        set => SetValue(HeaderProperty, value);
    }

    public static readonly DependencyProperty HeaderProperty =
        DependencyProperty.Register(nameof(Header), typeof(string), typeof(SettingCard), new PropertyMetadata(null));

    /// <summary>Secondary caption beneath the header; supports a OneWay-bound, changing value.</summary>
    public string Description
    {
        get => (string)GetValue(DescriptionProperty);
        set => SetValue(DescriptionProperty, value);
    }

    public static readonly DependencyProperty DescriptionProperty =
        DependencyProperty.Register(nameof(Description), typeof(string), typeof(SettingCard), new PropertyMetadata(null));
}
