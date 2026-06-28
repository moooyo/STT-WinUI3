using System.Text.RegularExpressions;

namespace Stt.Core.Text;

/// <summary>
/// Strips SenseVoice rich-transcription tags of the form <c>&lt;|...|&gt;</c> — language, emotion,
/// event, and ITN markers like <c>&lt;|zh|&gt;</c>, <c>&lt;|NEUTRAL|&gt;</c>, <c>&lt;|woitn|&gt;</c>
/// (spec §8.4) — returning the clean text plus the tag contents that were removed.
/// </summary>
public static partial class SpecialTagStripper
{
    [GeneratedRegex(@"<\|([^|>]*)\|>", RegexOptions.Compiled)]
    private static partial Regex TagRegex();

    public static (string Clean, IReadOnlyList<string> Tags) Strip(string text)
    {
        var tags = new List<string>();
        string clean = TagRegex().Replace(text, m =>
        {
            tags.Add(m.Groups[1].Value);
            return string.Empty;
        });
        return (clean.Trim(), tags);
    }
}
