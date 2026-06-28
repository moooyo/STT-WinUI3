using Microsoft.UI.Dispatching;
using Stt.Abstractions.Common;

namespace Stt.App.Services;

/// <summary>
/// <see cref="IUiDispatcher"/> backed by the WinUI <see cref="DispatcherQueue"/> (spec §6). All
/// partial/final updates from the engine are marshalled to the UI thread through here, keeping
/// <c>Stt.Core</c> free of any <c>Microsoft.UI.*</c> reference. Teardown-safe: a failed enqueue
/// (queue shutting down) is ignored.
/// </summary>
public sealed class DispatcherQueueUiDispatcher : IUiDispatcher
{
    private readonly DispatcherQueue _dispatcher;

    public DispatcherQueueUiDispatcher(DispatcherQueue dispatcher) => _dispatcher = dispatcher;

    public void Enqueue(Action action)
    {
        // Ignore the bool result: false means the queue is shutting down (spec §6 teardown).
        _dispatcher.TryEnqueue(() => action());
    }
}
