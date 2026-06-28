namespace Stt.Core.Models;

/// <summary>
/// Result of load-time model validation (spec §10.2). Lists every check performed (so a rejection
/// can show what was inspected) and any errors. <see cref="Ok"/> is true only when there are no
/// errors.
/// </summary>
public sealed class ValidationReport
{
    public List<string> Checks { get; } = new();
    public List<string> Errors { get; } = new();

    public bool Ok => Errors.Count == 0;

    public void Check(string message) => Checks.Add(message);
    public void Fail(string message) => Errors.Add(message);

    public override string ToString()
    {
        var lines = new List<string>();
        lines.Add(Ok ? "VALID" : "REJECTED");
        foreach (var c in Checks) lines.Add($"  [checked] {c}");
        foreach (var e in Errors) lines.Add($"  [ERROR]   {e}");
        return string.Join(Environment.NewLine, lines);
    }
}
