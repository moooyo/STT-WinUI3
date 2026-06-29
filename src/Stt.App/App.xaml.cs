using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.UI.Dispatching;
using Microsoft.UI.Xaml;
using Stt.Abstractions.Common;
using Stt.Abstractions.Models;
using Stt.App.Services;
using Stt.App.ViewModels;
using Stt.Core.Ep;
using Stt.Core.Models;

namespace Stt.App;

/// <summary>
/// Application entry point and composition root (spec §12). Builds a generic host with DI: the
/// model registry, EP selector, UI dispatcher, and per-page view models. The engine and ORT
/// sessions are owned here; page view models are transient.
/// </summary>
public partial class App : Application
{
    public static IHost Host { get; private set; } = null!;
    public static Window? MainWindowInstance { get; private set; }
    private Window? _window;

    public App()
    {
        InitializeComponent();
    }

    protected override void OnLaunched(LaunchActivatedEventArgs args)
    {
        // The UI dispatcher must be captured on the UI thread.
        var uiDispatcher = new DispatcherQueueUiDispatcher(DispatcherQueue.GetForCurrentThread());

        // Register the Windows ML certified EPs (DirectML + vendor NPU) in the background so
        // OrtEpEnumerator surfaces them; CPU works regardless if this is slow/unavailable (spec §9).
        _ = RegisterExecutionProvidersAsync();

        Host = Microsoft.Extensions.Hosting.Host
            .CreateDefaultBuilder()
            .ConfigureServices(services => ConfigureServices(services, uiDispatcher))
            .Build();

        _window = new MainWindow();
        MainWindowInstance = _window;
        _window.Activate();
    }

    private static async Task RegisterExecutionProvidersAsync()
    {
        try
        {
            await Microsoft.Windows.AI.MachineLearning.ExecutionProviderCatalog
                .GetDefault().EnsureAndRegisterCertifiedAsync();
        }
        catch { /* no catalog / offline / unsupported OS — engine runs on CPU + any built-in DML */ }
    }

    private static void ConfigureServices(IServiceCollection services, IUiDispatcher uiDispatcher)
    {
        string localAppData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        string modelsRoot = Path.Combine(localAppData, "Stt", "models");
        string cacheRoot = Path.Combine(localAppData, "Stt", "cache");
        Directory.CreateDirectory(modelsRoot);
        Directory.CreateDirectory(cacheRoot);

        services.AddSingleton(uiDispatcher);
        services.AddSingleton<IModelRegistry>(_ => new ModelRegistry(modelsRoot));
        services.AddSingleton(_ => new CompiledModelCache(cacheRoot));
        services.AddSingleton<IExecutionProviderSelector>(sp =>
            new ExecutionProviderSelector(cache: sp.GetRequiredService<CompiledModelCache>()));

        services.AddSingleton(_ => SttOptions.Load());
        services.AddSingleton<TranscriptionService>();

        services.AddTransient<MainViewModel>();
        services.AddTransient<ModelManagerViewModel>();
        services.AddTransient<SettingsViewModel>();
    }

    /// <summary>Resolve a service from the host container (for code-behind that can't inject).</summary>
    public static T GetService<T>() where T : notnull => Host.Services.GetRequiredService<T>();
}
