using System.Text;

namespace Stt.Core.Text;

/// <summary>
/// Reconstructs text from SentencePiece pieces (spec §8.4). The meta-symbol <c>▁</c> (U+2581)
/// marks a word boundary (a leading space); CJK characters are emitted as standalone pieces with
/// no boundary marker, so they concatenate directly. Example:
/// <c>["▁hello","▁world"] → "hello world"</c>; <c>["中","文","▁mix","ed"] → "中文 mixed"</c>.
/// </summary>
public static class SentencePieceDetokenizer
{
    private const char Boundary = '▁'; // ▁

    public static string Decode(IEnumerable<string> pieces)
    {
        var sb = new StringBuilder();
        foreach (string piece in pieces)
            sb.Append(piece);

        // Replace boundary markers with spaces, collapse, and trim.
        sb.Replace(Boundary, ' ');
        return sb.ToString().Trim();
    }
}
