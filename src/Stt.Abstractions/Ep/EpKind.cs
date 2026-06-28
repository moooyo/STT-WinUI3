namespace Stt.Abstractions.Ep;

/// <summary>Execution provider kinds exposed through Windows ML / ORT (spec §5.2).</summary>
public enum EpKind
{
    /// <summary>CPU EP — always available (Win10 1809+).</summary>
    Cpu,

    /// <summary>DirectML — GPU, Win10 1809+.</summary>
    DirectML,

    /// <summary>CUDA — NVIDIA GPU.</summary>
    Cuda,

    /// <summary>QNN — Qualcomm NPU. No dynamic shapes; needs Win11 24H2+.</summary>
    Qnn,

    /// <summary>OpenVINO — Intel CPU/GPU/NPU.</summary>
    OpenVINO,

    /// <summary>VitisAI — AMD/Xilinx NPU.</summary>
    VitisAI
}
