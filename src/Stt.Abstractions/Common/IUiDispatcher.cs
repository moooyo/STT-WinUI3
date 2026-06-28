namespace Stt.Abstractions.Common;

/// <summary>
/// Abstraction over the UI thread's dispatcher (spec §5.1, §6). The Core engine marshals
/// partial/final updates to the UI through this interface so it never references
/// <c>Microsoft.UI.*</c>. The app implements it by wrapping <c>DispatcherQueue.TryEnqueue</c>;
/// headless tests implement it as an inline/queued invoker.
/// </summary>
public interface IUiDispatcher
{
    /// <summary>Marshal an action onto the UI thread. Implementations must not throw on teardown.</summary>
    void Enqueue(Action action);
}
