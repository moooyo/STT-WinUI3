using Stt.Core.Text;

namespace Stt.Core.Tests.Text;

public class TokenTableTests
{
    [Fact]
    public void Parses_Piece_Id_Format_Bidirectional()
    {
        var t = TokenTable.Parse(new[] { "<blk> 0", "▁the 1", "中 2" });
        Assert.Equal(3, t.Count);
        Assert.Equal("▁the", t.Piece(1));
        Assert.True(t.TryId("中", out int id));
        Assert.Equal(2, id);
        Assert.False(t.TryId("missing", out _));
    }

    [Fact]
    public void Falls_Back_To_Line_Index_When_No_Id_Column()
    {
        var t = TokenTable.Parse(new[] { "a", "b", "c" });
        Assert.Equal("b", t.Piece(1));
        Assert.True(t.TryId("c", out int id));
        Assert.Equal(2, id);
    }

    [Fact]
    public void Out_Of_Range_Id_Returns_Empty()
    {
        var t = TokenTable.Parse(new[] { "a 0" });
        Assert.Equal(string.Empty, t.Piece(99));
    }
}

public class SentencePieceDetokenizerTests
{
    [Fact]
    public void Joins_Word_Pieces_With_Spaces()
    {
        Assert.Equal("hello world", SentencePieceDetokenizer.Decode(new[] { "▁hello", "▁world" }));
    }

    [Fact]
    public void Mixes_Cjk_And_Latin()
    {
        Assert.Equal("中文 mixed", SentencePieceDetokenizer.Decode(new[] { "中", "文", "▁mix", "ed" }));
    }
}

public class SpecialTagStripperTests
{
    [Fact]
    public void Removes_SenseVoice_Tags_And_Returns_Them()
    {
        var (clean, tags) = SpecialTagStripper.Strip("<|zh|><|NEUTRAL|>你好<|woitn|>");
        Assert.Equal("你好", clean);
        Assert.Contains("zh", tags);
        Assert.Contains("NEUTRAL", tags);
        Assert.Contains("woitn", tags);
    }

    [Fact]
    public void Leaves_Plain_Text_Untouched()
    {
        var (clean, tags) = SpecialTagStripper.Strip("hello world");
        Assert.Equal("hello world", clean);
        Assert.Empty(tags);
    }
}
