namespace Stt.Core.Text;

/// <summary>
/// A bidirectional token table loaded from a <c>tokens.txt</c> sidecar (spec §10.1: tokens are
/// always a sidecar; the ONNX graph carries only <c>vocab_size</c>). The canonical format is one
/// <c>"&lt;piece&gt; &lt;id&gt;"</c> entry per line (sherpa/icefall/FunASR). Files that list only
/// the piece per line are accepted with the id taken from the line index.
/// </summary>
public sealed class TokenTable
{
    private readonly string[] _idToPiece;
    private readonly Dictionary<string, int> _pieceToId;

    private TokenTable(string[] idToPiece, Dictionary<string, int> pieceToId)
    {
        _idToPiece = idToPiece;
        _pieceToId = pieceToId;
    }

    public int Count => _idToPiece.Length;

    public static TokenTable Load(string tokensPath) => Parse(File.ReadAllLines(tokensPath));

    public static TokenTable Parse(IEnumerable<string> lines)
    {
        var map = new SortedDictionary<int, string>();
        int lineIndex = 0;
        foreach (string raw in lines)
        {
            string line = raw.TrimEnd('\r', '\n');
            if (line.Length == 0) { lineIndex++; continue; }

            // Split off the trailing id; the remainder (which may itself contain spaces) is the piece.
            int sep = line.LastIndexOf(' ');
            string piece;
            int id;
            if (sep > 0 && int.TryParse(line.AsSpan(sep + 1), out id))
                piece = line[..sep];
            else
            {
                piece = line;
                id = lineIndex;
            }
            map[id] = piece;
            lineIndex++;
        }

        if (map.Count == 0) return new TokenTable(Array.Empty<string>(), new());

        int max = 0;
        foreach (int k in map.Keys) max = Math.Max(max, k);
        var arr = new string[max + 1];
        var rev = new Dictionary<string, int>(map.Count);
        foreach (var kv in map)
        {
            arr[kv.Key] = kv.Value;
            rev[kv.Value] = kv.Key;
        }
        return new TokenTable(arr, rev);
    }

    public string Piece(int id) =>
        id >= 0 && id < _idToPiece.Length ? _idToPiece[id] ?? string.Empty : string.Empty;

    public bool TryId(string piece, out int id) => _pieceToId.TryGetValue(piece, out id);
}
