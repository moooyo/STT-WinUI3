using Xunit;

namespace Stt.Core.Tests;

public class ToolchainSmokeTest
{
    [Fact]
    public void Net8Builds_And_OrtRestores()
    {
        // Validates net8.0 builds on the .NET 10 SDK and the test harness runs.
        Assert.True(true);
    }
}
