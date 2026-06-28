# 本地 STT 引擎设计 (Local Streaming Speech-to-Text Engine)

- **日期**: 2026-06-28
- **状态**: Draft (待评审)
- **目标平台**: Windows 桌面 (WinUI3 / Windows App SDK, .NET 8 → 已迁移至 **.NET 10 + Windows App SDK 2.2**，见 §4 基线)
- **范围**: 完整的可插拔两遍 (two-pass) 本地语音转文字系统，运行时为裸 ONNX Runtime / Windows ML，跨硬件 (CPU / GPU / NPU)，多语言 (首要中英混读)。

> 说明：本仓库目录名为 `tts-winui3`，但真实目标是**本地语音转文字 (STT)**，不是 TTS。

---

## 1. 概述、目标与非目标

### 1.1 目标
- 本地、离线、隐私安全的实时语音转文字。
- **多语言 + 中英混读** (intra-sentence code-switching)：一段中文里夹英文单词/短语。
- **两遍解码 (模式②)**：流式首遍出实时局部字幕；说完一句 (VAD 端点) 后用更强的离线模型对整段**重识别 (re-decode)** 产出终稿，替换局部结果。
- **可插拔**：用户自选模型、自选一遍/两遍；新模型家族可按需扩展。
- **跨硬件**：经 Windows ML 的 Execution Provider，单一 ONNX 在 CPU / GPU(DirectML) / NPU 上运行。先 CPU/GPU，NPU 预留。

### 1.2 非目标 (v1)
- 不做云端 ASR (Qwen3-ASR-Flash 等)；仅本地。可保留"云端兜底"作为未来扩展点，不在 v1。
- 不做说话人分离 (diarization)、标点/ITN 之外的后处理 (SenseVoice 自带 ITN 可用)。
- 不做模型训练/微调；只消费已导出的 ONNX 模型。
- Whisper 自回归解码 (onnxruntime-genai) **不在 v1 核心**，作为可选插件 (见 §8.5)。

---

## 2. 决策记录 (Locked Decisions)

| # | 决策 | 状态 |
|---|---|---|
| D1 | 运行时 = 裸 ONNX Runtime / **Windows ML** (`Microsoft.WindowsAppSDK.ML`)，**不用 sherpa-onnx NuGet** | 已定 |
| D2 | 两遍解码 = **模式② (re-decode)**：流式首遍 + 句末独立离线模型重识别 | 已定 |
| D3 | 流式首遍 = **方案 A：流式 Zipformer transducer** (encoder+decoder+joiner) | 已定 |
| D4 | 离线第二遍默认 = **SenseVoice-Small** (NAR/CTC)，且**可由用户更换** | 已定 |
| D5 | 架构 = **方案 1：分层 + 策略接口** (UI 无关的 Core 引擎库 + WinUI App) | 已定 |
| D6 | 特征提取允许**原生 shim**：P/Invoke `kaldi-native-fbank` (bit 兼容) | 已定 |
| D7 | 模型分发 = **用户自带** (侧载文件夹 + manifest/自动识别)，不内置默认模型 | 已定 |
| D8 | UI = **完整应用** (转写窗 + 模型管理器 + 管线/设置) | 已定 |
| D9 | 打包 = **unpackaged 自包含** (默认)；MSIX 作后续变体 | **待评审确认 (见 §13)** |
| D10 | NPU 预留：先 CPU/DirectML；NPU 只规划给定形非自回归编码器 | 已定 |

---

## 3. 范围与分阶段

一份 spec 覆盖完整系统设计，但**实现分阶段**：

- **Phase 0 — 离线单遍 + 基础设施**：mic → VAD → kaldi-fbank → SenseVoice (ORT/Windows ML) → 文本。打通 Windows ML 包/EP 选择/编译缓存/容错骨架 + 特征 golden test。交付一个"说完一句出中英混读文本"的可用产品。
- **Phase 1 — 流式首遍 → 两遍**：实现流式 Zipformer transducer 解码 + 状态机 harness (先对齐 sherpa 参考输出)，接入两遍模式。
- **Phase 2 — GPU**：DirectML EP 加速 (尤其离线第二遍编码器)。
- **Phase 3 — NPU**：定形量化的非自回归/流式编码器经 Windows ML 下发 QNN/OpenVINO/VitisAI；需 Win11 24H2+。

---

## 4. 解决方案 / 项目结构

```
Stt.sln
├─ src/
│  ├─ Stt.Abstractions      (net8.0)            接口 + DTO + 枚举，零三方依赖
│  ├─ Stt.Core              (net8.0)            引擎；依赖 ORT/Channels；不引用 Microsoft.UI.*
│  │     Audio/        RingBuffer, Resampler, FrameChunker, FileAudioCapture(测试用)
│  │     Features/     KaldiFbankFrontend, WhisperMelFrontend, Lfr, Cmvn, FeatureFamilyDetector
│  │     Vad/          SileroVad
│  │     Decoders/     TransducerDecoder, CtcDecoder, NarDecoder, EncoderStateFactory
│  │     Ep/           ExecutionProviderSelector, SessionOptionsBuilder, CompiledModelCache
│  │     Models/       ModelManifest, ModelRegistry, ModelLoader, capability flags
│  │     Pipeline/     SttPipeline, PipelineMode, EndpointDetector
│  ├─ Stt.Audio.Windows     (net8.0-windows)    IAudioCapture 的 WASAPI/NAudio 实现
│  └─ Stt.App               (net8.0-windows10.0.19041.0, WinUI3, Windows App SDK)
│        App.xaml.cs (Host+DI), Views/, ViewModels/, Converters/, Services/UiDispatcher
└─ tests/
   ├─ Stt.Core.Tests        (net8.0, xUnit)     headless: WAV → 断言文本；特征 golden；解码对齐
   └─ Stt.Pipeline.Tests    (net8.0)            channel 背压/取消/生命周期
```

**分层规则**：依赖单向 `App → Core/Audio → Abstractions`；`Stt.Core` 永不引用 `Microsoft.UI.*`，因而可 headless 单测。`Stt.Audio.Windows` 隔离唯一 OS 绑定的音频依赖，使 Core 测试用 `FileAudioCapture` 喂 WAV 跑完整识别链路 (CI 无麦无 UI)。`Stt.App` 是唯一引用 Windows App SDK 的项目。

原生依赖 (`onnxruntime.dll`、`onnxruntime-genai`、`kaldi-native-fbank.dll`) 按 RID 放 `runtimes/win-x64|win-arm64/native`，发布时落到 exe 旁。

**目标框架基线**：构建 TFM `net8.0-windows10.0.19041.0`；NPU/优化 EP 代码路径运行时门控在 build 26100 (24H2)，以下回退 DirectML/CPU。架构仅 x64 + ARM64 (ARM64 关乎 Copilot+ / Snapdragon NPU)。

> **更新 (2026-06-28，实现后)**：应用户要求已整体迁移到 **.NET 10**——库/测试为 `net10.0`，`Stt.Audio.Windows` 为 `net10.0-windows`，`Stt.App` 为 `net10.0-windows10.0.19041.0` 并升级到 **Windows App SDK 2.2**（自 1.6）。CommunityToolkit `SettingsControls`（尚无 Windows App SDK 2.x 版本）已移除，改用原生 `SettingCard` + 内置 `Expander`；仅保留与 UI 框架无关的 `CommunityToolkit.Mvvm`。`TargetPlatformMinVersion`/运行时 26100 门控不变。

---

## 5. 核心抽象 (可插拔)

### 5.1 接口 (Stt.Abstractions)

```csharp
public interface IAudioCapture : IDisposable {
    event Action<AudioFrame> FrameAvailable;   // 16k mono float
    Task StartAsync(CancellationToken ct);
    Task StopAsync();
}

public interface IFeatureFrontend {
    AsrFeatureFamily Family { get; }
    int FeatureDim { get; }
    // 返回 [numFrames, FeatureDim] 行优先，按模型确切约定
    float[] Extract(ReadOnlySpan<float> pcm16kMono, out int numFrames);
}

public interface IVad : IDisposable {
    void Reset();
    void AcceptWaveform(ReadOnlySpan<float> window512);  // 512 @16k
    bool TryDequeueSegment(out SpeechSegment seg);
}

public interface IAsrDecoder : IDisposable {
    DecoderCapabilities Capabilities { get; }
    void Reset();
    bool AcceptFeatures(ReadOnlySpan<float> features, int numFrames, int featDim);
    void InputFinished();
    bool IsEndpoint();
    AsrResult GetResult();
}

public interface IExecutionProviderSelector {
    // 构建带选定 EP 的 SessionOptions；内部处理编译缓存与回退
    SessionOptions BuildSessionOptions(EpPreference pref, string modelHash);
}

public interface IModelRegistry {
    IReadOnlyList<ModelManifest> List();
    ModelManifest Get(string id);
    ModelManifest ImportFromFolder(string folderPath);   // 侧载 + 自动识别
    void Remove(string id);
}

public interface ISttPipeline : IAsyncDisposable {
    PipelineConfig Config { get; }
    event Action<PartialResult> Partial;   // 实时局部 (灰字)
    event Action<FinalResult> Final;       // 终稿 (黑字替换)
    Task StartAsync(CancellationToken ct);
    Task StopAsync();
}

public interface IUiDispatcher { void Enqueue(Action action); }  // 抽象 DispatcherQueue，保持 Core 无 UI
```

### 5.2 DTO / 枚举

```csharp
public sealed record AudioFrame(float[] Samples, int Count);          // 来自 ArrayPool
public sealed record SpeechSegment(long StartSample, float[] Samples);
public sealed record AsrResult(string Text, IReadOnlyList<int> Tokens,
                               IReadOnlyList<float> Timestamps, bool IsFinal);
public sealed record PartialResult(int SegmentId, string Text);
public sealed record FinalResult(int SegmentId, string Text);

public enum AsrFeatureFamily { Auto, KaldiFbankPovey, KaldiFbankLfrCmvn,
                               WhisperLogMel, NemoMel, Mfcc, RawAudioSamples }
public enum DecoderType { Transducer, Ctc, Nar, Ar }
public enum PipelineMode { OnePassStreaming, OnePassOffline, TwoPass }
public enum EpKind { Cpu, DirectML, Cuda, Qnn, OpenVINO, VitisAI }

[Flags] public enum DecoderCapabilities {
    None = 0, Streaming = 1, Offline = 2, PartialResults = 4,
    Endpointing = 8, Timestamps = 16, Multilingual = 32 }
```

### 5.3 模型描述符 (manifest)

多数字段从 ONNX `metadata_props` 自动派生 (见 §10)；缺失才由用户/侧载补。

```jsonc
{
  "id": "zipformer-zh-en-streaming",
  "displayName": "Zipformer 中英(流式)",
  "version": "1.0.0",
  "family": "transducer",                 // transducer|paraformer|zipformer2_ctc|whisper|sense_voice|...
  "runtime": ["streaming"],               // streaming | offline | 两者(hybrid)
  "decoderType": "transducer",            // transducer|ctc|nar|ar
  "files": { "encoder":"encoder.onnx", "decoder":"decoder.onnx",
             "joiner":"joiner.onnx", "tokens":"tokens.txt" },
  "feature": { "frontEnd":"kaldi_fbank", "family":"KaldiFbankPovey",
               "sampleRate":16000, "featureDim":80, "lfr":null, "cmvn":"none" },
  "capabilities": { "streamingCapable":true, "offlineCapable":false,
                    "needsLfrCmvn":false, "multilingual":true,
                    "emitsTimestamps":true, "needsVad":false },
  "languages": ["zh","en"],
  "decoding": { "defaultMethod":"greedy_search",
                "endpointRules": { "rule2MinTrailingSilence":1.2 } },
  "providerSupport": ["cpu","directml"],  // 动态形状的省略 qnn
  "license": "Apache-2.0"
}
```

**能力标志驱动合法组合** (UI 校验)：首遍槽只接受 `streamingCapable==true` → "Whisper 不能当首遍"自动成立；2 遍要求有 VAD。

---

## 6. 数据流与线程模型

三个独立"世界" (音频回调线程 / 推理 worker / UI 线程) 经 `System.Threading.Channels` 解耦。UI 线程永不阻塞、永不推理；音频线程永不分配、永不调用 ORT。

```
[音频回调线程] 复制设备缓冲→ArrayPool<float>→(需要则重采样16k mono)→Writer.TryWrite(frame)  非阻塞
      │  Channel<AudioFrame>  有界, SingleWriter/SingleReader, FullMode=DropOldest   ← 背压#1
      ▼
[首遍 worker  (TaskCreationOptions.LongRunning, 独立 streaming session)]
      await foreach frame: VAD/分块累积 → fbank → transducer/ctc 增量解码 → 局部假设
      → uiDispatcher.Enqueue(set PartialText)  (合并 ~100–150ms)
      → 端点(VAD 静音/最大时长)时把整段 → segmentCh.Writer
      │  Channel<SpeechSegment>  有界, FullMode=Wait(终稿不丢)   ← 背压#2
      ▼
[第二遍 worker (LongRunning, 低优先, 独立 offline session)]
      await foreach seg: SenseVoice 整段重识别 → uiDispatcher.Enqueue(ReplaceSegment(id, finalText))
      ▼
[UI 线程 DispatcherQueue] 只 set ObservableProperty
```

**关键约束**：
- 音频回调用 `TryWrite` (同步非阻塞)；满则按 DropOldest 丢旧帧 (实时优先) 并经 `itemDropped` 回调计数 → UI "落后"指示。**背压#1 绝不用 Wait** (会阻塞实时音频回调)。
- 背压#2 用 `WriteAsync` + Wait (终稿珍贵，首遍 worker 可 await，它不是实时线程)。
- **每 worker 独立 `InferenceSession`** (不跨线程共享同一 session 的并发 Run，规避 ORT 线程安全不确定性)；共享 ORT 全局 intra-op 线程池并限核，避免过订阅。
- UI marshal 用 `IUiDispatcher` 抽象 → 实现包 `DispatcherQueue.TryEnqueue`；高频局部合并发送。
- per-session `CancellationTokenSource` 贯穿 `ReadAllAsync(ct)` / Run 循环 / WriteAsync；Stop 为 `AsyncRelayCommand` await worker 完成；ORT 对象确定性 Dispose；teardown 忽略 `TryEnqueue` 返回 false 与晚到写的 `ChannelClosedException`。

---

## 7. 特征子系统 (最高风险)

特征不匹配 = 静默乱码 (编码器照常出有限 logits，解码器吐"像样但错误"的字，无异常)。根治策略：**按家族选对前端 + 参数从 metadata 读 + 缺参硬报错 + golden test**。

### 7.1 家族枚举与实现折叠
6 个家族 (`AsrFeatureFamily`)，实际只需 **4 提取器 + 2 后处理**：

| 家族 | 引擎 | 窗/mel/log | 维度 & 布局 |
|---|---|---|---|
| A KaldiFbankPovey | icefall Zipformer/CTC/tdnn/wenet_ctc | povey/HTK mel/自然log/无CMVN | 80 `[N,T,80]` |
| B KaldiFbank+LFR+CMVN | FunASR Paraformer/SenseVoice | hamming/+LFR(7)+CMVN | 560=80×7 `[N,T,560]` |
| C WhisperLogMel | Whisper/Dolphin/FireRedASR | Slaney/Hann/log10/固定30s | 80或128 `[N,n_mels,3000]` |
| D NemoMel | NeMo/GigaAM | =kaldi-fbank(Slaney+Hann) + 逐特征归一化 | 80/128/64 `[N,T,C]` |
| E Mfcc | TeleSpeech/TDNN | is_mfcc/num_ceps=40 | 40 |
| F RawAudio | T-one | 无 fbank (原始 PCM) | 1-D |

实现：**`KaldiFbankFrontend` (可配 window/is_librosa/low-high_freq/dither/snip_edges/normalize_samples + 可选 LFR/CMVN 后处理) 覆盖 A/B/D**；`WhisperMelFrontend` (载 OpenAI `mel_filters.npz`) 覆盖 C；`Mfcc`、`RawAudio` 各一。后处理 `Lfr`、`Cmvn`、`PerFeatureNorm`。**v1 实现 A + B** (默认两遍只用这俩)。

### 7.2 实现选择
- **`KaldiFbankFrontend` = P/Invoke `kaldi-native-fbank`** (Apache-2.0, 无依赖, win-x64/arm64, 与训练 bit 兼容)。配 80 bins、dither=0、snip_edges=false、按 metadata 设 normalize_samples。
- LFR + CMVN 用 C# 几行：stack `lfr_m` 帧步进 `lfr_n` → 560；`x=(x+neg_mean)*inv_stddev`。常量从 ONNX metadata 读。
- Whisper 前端：载 `mel_filters.npz` (不自重算滤波器)，managed STFT + log10/clamp/affine；或导出时用 `onnxruntime-extensions` 把 mel 烤进图。

### 7.3 验证 (CI 强制)
对每个家族对 Python 权威工具 (lhotse / funasr_onnx / whisper) 逐元素 diff，断言 **max-abs < 1e-3, mean-abs < 1e-4**。诊断：常数偏移≈10.4=幅度(×32768)开关错；随幅度缩放=mel/power 错；仅帧边=snip_edges/padding 错。作为 golden CI test。

---

## 8. 解码器子系统

### 8.1 统一接口
`IAsrDecoder` (见 §5.1)：流式增量解码、离线 buffer 到 `InputFinished()` 再解。`IsEndpoint()` 仅对流式有意义。各家族的解码循环/状态都是**实现内部细节**，不进接口。

### 8.2 流式 Zipformer transducer (D3)
- 3 个 ONNX 图：encoder + decoder(predictor) + joiner。
- **状态张量通用零初始化** (从 metadata 派生，不硬编码)：读 `encoder_dims/query_head_dims/value_head_dims/num_heads/num_encoder_layers/cnn_module_kernels/left_context_len` (逗号数组) + `T/decode_chunk_len/context_size/vocab_size`。状态总数 `m×6+2` (m=Σnum_encoder_layers)：每层 6 个 cache (key/nonlin_attn/val1/val2/conv1/conv2，按公式推维) + 全局 embed_states + processed_lens(int64)。
- 贪心解码：逐 encoder 输出帧 → joiner → argmax → 非 blank(id=0) 则 append 并重跑 predictor 刷新 decoder_out；blank 则复用。缓存 decoder_out 跨 chunk。modified_beam_search 为同机制加 beam (后续)。
- **参考移植** sherpa `csrc/online-zipformer2-transducer-model.cc` (状态/RunEncoder/Run*) + `online-transducer-greedy-search-decoder.cc` + icefall `export-onnx-streaming.py` (张量名/形/顺序权威)。
- **数值对齐**：先把 chunked 状态 harness 对齐 sherpa-onnx 同模型输出，再信任。

### 8.3 流式 CTC (可选简化首遍)
单图 encoder → `log_probs`；解码 ~20 行 argmax + collapse，跨 chunk 仅 `prev_id`。状态机与 transducer 共享。作为低资源/NPU 友好备选；中英流式 CTC 模型可用性需核实，否则用 transducer。

### 8.4 NAR 离线 (第二遍, D4)
SenseVoice / Paraformer：单次 `Run` → argmax/CTC collapse。SenseVoice：fbank80→LFR(7,6)→CMVN→encoder+CTC→logits，剥前 4 个 language/event/emotion/textnorm query 槽，SentencePiece 反token，剥 `<|...|>` 标签；语言/ITN token id 从 metadata 读。按长度桶 padding。

### 8.5 Whisper-AR 经 genai (延后, 可选插件)
`Microsoft.ML.OnnxRuntimeGenAI` 内部处理自回归循环 + KV cache + beam，C# `Model/Generator/MultiModalProcessor/Audios`。**不在 v1 核心**：它自带 ORT/EP (不继承应用统一的 Windows ML EP 选择，需经 `Config.AppendProvider` 翻译)、按 EP 带平行原生包、离线/30s 分块不进流式热循环、C# API 仍 preview。接口预留 `Ar` 能力 → 将来 `WhisperGenAiDecoder` drop-in (`Microsoft.ML.OnnxRuntimeGenAI.WinML` 可跑在 Windows ML 共享运行时上)。

### 8.6 热循环性能
IO binding (`OrtIoBinding`) + 预分配 `OrtValue` (`CreateTensorValueFromMemory` / `GetTensorMutableDataAsSpan`) + **状态双缓冲** (A 入 B 出后交换引用，因 ORT 不能同缓冲读写)。每 chunk 只拷新特征入预分配 buffer，不新建 OrtValue。session 结束统一 Dispose 解钉。

---

## 9. Execution Provider / Windows ML

- **包**：`Microsoft.WindowsAppSDK.ML` (或 `Microsoft.Windows.AI.MachineLearning`) 作 ORT 提供者；**移除独立 `Microsoft.ML.OnnxRuntime` 以免双加载 `onnxruntime.dll`**。启动后台线程 `ExecutionProviderCatalog.GetDefault().EnsureAndRegisterCertifiedAsync()` (带进度 UI)。
- **EP 选择**：显式 `OrtEnv.GetEpDevices()` 按 `EpName`+`HardwareDevice.Type` 过滤 `AppendExecutionProvider` (每次建 session 重枚举，别缓存设备列表)；或策略 `SetEpSelectionPolicy(MAX_EFFICIENCY/PREFER_NPU/...)`。建 session try/catch 降级 CPU。
- **一切定形 (铁律)**：DirectML 动态轴慢约 5×。流式编码器定 chunk + 定 cache；离线定窗或按长度桶 padding；用 `make_dynamic_shape_fixed` / `AddFreeDimensionOverrideByName`。
- **EPContext 编译缓存**：`OrtModelCompilationOptions` → `_ctx.onnx` 写 `LocalCacheFolder`，文件名打戳 `{modelHash}_{epName}_{epVer}_{driver}`；加载 `INVALID_GRAPH` → 删旧 + 后台重编 + 期间跑 CPU。把"重编"当常态非致命。
- **组件 → EP**：流式编码器 = DirectML/CPU (唯一可上 NPU 的 ASR 部件，需全静态+量化+无 Loop/If)；transducer decoder/joiner = CPU (小矩阵)；离线 NAR 编码器 = NPU(静态)/DirectML；自回归解码器 = CPU/GPU。
- 门槛：CPU+DirectML 全 Win10 1809+；NPU/优化 GPU EP 需 Win11 24H2 (26100)+。

---

## 10. 模型管理与加载校验

### 10.1 自动识别
读 `InferenceSession.ModelMetadata.CustomMetadataMap` (Dictionary<string,string>) + `InputMetadata` 输入张量维度：
- `model_type` 派发家族 (zipformer2/paraformer/sense_voice_ctc/whisper-*/*_ctc/EncDec*);
- **输入特征维度是仲裁信号**：80=fbank 或 whisper-80 (歧义靠 model_type/n_mels/布局二分)；128=whisper-large-v3；560=FunASR；布局 fbank/LFR=`[N,T,C]`(末轴/时间动态)，Whisper=`[N,mels,3000]`(中轴/时间固定3000)。
- FunASR 四键 (lfr_window_size/shift, neg_mean, inv_stddev) + normalize_samples + 语言/ITN token；transducer 状态几何 (§8.2)。
- tokens.txt 永远 sidecar (图里只 vocab_size 计数)。

### 10.2 五层 fail-loud 校验 (加载时)
1. 自动识别家族 (上)。
2. 校验用户选择：选的家族 expectedDim ≠ 编码器实际输入维 → **拒** (输入维是仲裁者；用户只能标注歧义、不可违背图)。
3. 必需参数闸门：B 缺 lfr/cmvn → 拒，断言 `80×lfr == featDim`；C 的 n_mels ≠ 维 → 拒。
4. tokens 非空行数 ≠ vocab_size → 拒 (配错 tokens)。
5. 可选自检：内置 WAV → 前端 → 编码器前向，特征范围 sanity (fbank 自然log≈[-20,20]，whisper≈[-1,1]) + 输出非空非 NaN。

**铁律**：识别为 Unknown 就停下问，绝不默认 fbank-80；缺必需参数硬拒；重导出可能把特征轴标动态(-1)→读不到则要求手动指定。

### 10.3 注册表与侧载
内置目录可空 (D7 用户自带)；扫 `LocalAppData/models/` 与用户导入路径：有 `manifest.json` 就载；没有就从文件命名 (encoder/decoder/joiner=transducer；单 model.onnx=ctc/paraformer) + ONNX metadata 推临时 manifest 让用户确认。外来 (裸 Whisper/Optimum/裸 NeMo) 无 metadata → 强制要求用户 manifest。注册表 API：`List/Get/ImportFromFolder/Remove/ResolveCombination(pass1,pass2,mode)`。

---

## 11. 管线模式

`PipelineMode`：
- **OnePassStreaming**：仅流式局部 (无终稿重识别)。
- **OnePassOffline**：VAD 切句 → 离线模型整段转写 (整句延迟，无逐词)。
- **TwoPass** (默认)：流式局部 + 句末离线重识别替换。

校验 (按能力标志)：首遍须 `streamingCapable`；2 遍/离线须有 VAD；2 遍第二遍须 `offlineCapable`；每遍各自前端 (不跨家族共享帧)。端点触发 = VAD 静音 (0.5–1.2s) **或** 最大时长上限 (防长句不停)。默认配置 = 流式 Zipformer zh-en + SenseVoice。

---

## 12. UI 设计 (完整应用)

WinUI3 + `CommunityToolkit.Mvvm` (`[ObservableProperty]` / `[RelayCommand]` / `AsyncRelayCommand` / `WeakReferenceMessenger`) + `Microsoft.Extensions.Hosting` DI (引擎为 `IHostedService`/`BackgroundService`；`InferenceSession` 持有者单例、页 VM 瞬态；`IOptions<SttOptions>` 配置)。

页面：
- **转写窗 (MainPage)**：实时局部 (灰/斜) + 终稿 (黑) 列表、录音 Start/Stop、设备/语言选择、"落后"指示、复制/导出。
- **模型管理器**：已安装列表 + 侧载导入 (FolderPicker) + 能力徽章 (streaming/offline/multilingual/int8) + 加载校验结果展示 + 删除。
- **管线与设置**：1/2 遍选择 + 每遍指派模型 (非法灰掉 + hover 原因) + EP 偏好 (Auto/CPU/GPU/NPU) + VAD/端点阈值。

UI 永不阻塞：所有局部/终稿更新经 `IUiDispatcher`。麦克风权限：`<DeviceCapability Name="microphone"/>` (packaged) 或全信任 (unpackaged)，拒绝时提示并停录。

---

## 13. 打包与部署 (§D9 待确认)

**默认 = unpackaged + 双自包含** (`WindowsAppSDKSelfContained=true` + .NET `SelfContained=true`)：
- 原因：本应用核心是**加载原生库 + 按路径读用户任意模型文件夹**。unpackaged 是全信任 Win32 进程 → 任意路径文件访问无需 `broadFileSystemAccess`/FutureAccessList，原生 DLL app-base 探测"即用"，无 MSIX 原生资源入包坑。
- 原生库按 RID 放 `runtimes/win-x64|win-arm64/native` (含自建 `kaldi-native-fbank.dll`)；每 RID 单独构建。
- 模型按路径就地读；**EPContext/硬件相关缓存写 `LocalCacheFolder`** (可再生、机器本地、不可跨 arch 漫游)。

**MSIX 变体 (后续 Store/企业渠道)**：用 FolderPicker + FutureAccessList 取用户文件夹真实路径喂原生加载器；缓存仍 `LocalCacheFolder`；验证各 arch 原生 DLL 确在 `.msix` 内；`broadFileSystemAccess` 仅在 picker 不够时启用 (需 Store 说明)。NAudio 在 MSIX Release 有裁剪坑 → 排除 `NAudio*` 裁剪并早测打包 Release。

> **评审决策点**：确认 v1 走 unpackaged，还是直接上 MSIX。

---

## 14. 错误处理

| 情形 | 处理 |
|---|---|
| 特征家族识别失败/不匹配 | 加载时**大声拒绝** (列出已检查项)，不静默跑错 |
| 必需参数缺失 | 硬拒，指出缺哪个 key |
| EP 不可用/设备运行时消失 | try/catch 降级 CPU；策略 PREFER_* 自带兜底 |
| 编译缓存失效 (EP/驱动更新) | INVALID_GRAPH → 删旧 + 后台重编 + 期间 CPU |
| 推理落后于音频 | DropOldest 丢旧帧 + UI "落后"指示 + 计数 |
| 麦克风被拒 | 捕获 UnauthorizedAccessException，提示 + 停录 |
| 网络受限 (EP catalog 下载失败) | 回退 in-box CPU/DirectML，标"加速受限"，稍后重试 |
| 取消/teardown | CTS 贯穿；忽略 TryEnqueue=false 与晚写 ChannelClosedException |

---

## 15. 测试策略

- **headless Core 测试** (xUnit, net8.0)：`FileAudioCapture` 喂 WAV → 断言转写文本，全链路无 UI/无麦在 CI 跑。
- **特征 golden test**：C# 前端 vs Python 参考逐元素 diff < 1e-3 (§7.3)。
- **解码对齐测试**：小样本 transducer/CTC/NAR 输出 vs sherpa-onnx 参考 1-best。
- **加载校验单测**：各家族正常识别 + 各类错配 (维度不符/缺参/tokens 不符) 被拒。
- **管线测试** (Stt.Pipeline.Tests)：channel 背压 (DropOldest 计数 / 第二遍 Wait)、取消、生命周期、`InputFinished` 终稿替换。
- **打包冒烟**：早期跑 unpackaged 自包含 Release，验证原生 DLL 加载 + 麦克风 + 任意路径读模型。

---

## 16. 分阶段实现路线

- **Phase 0**：项目骨架 (§4) + `IAudioCapture`(NAudio) + `IFeatureFrontend`(KaldiFbank+LFR/CMVN, P/Invoke) + 特征 golden + Silero VAD + NAR SenseVoice 解码 + `IExecutionProviderSelector`(CPU/DirectML + 编译缓存 + 回退) + 1-pass offline 管线 + 最小转写窗 + 模型加载校验。交付"说完出中英混读文本"。
- **Phase 1**：`EncoderStateFactory` (metadata 通用零初始化) + transducer 贪心解码 (对齐 sherpa) + 流式热循环 (IO binding/双缓冲) + 端点检测 + 2-pass 管线 + 实时局部字幕 UI。
- **Phase 2**：DirectML EP (第二遍编码器优先) + 多量化变体选择。
- **Phase 3**：定形量化非自回归/流式编码器经 Windows ML 下发 NPU；可选 Whisper-AR(genai) 插件。

---

## 17. 风险登记

| 风险 | 影响 | 缓解 |
|---|---|---|
| 特征不匹配 | 静默乱码 | P/Invoke kaldi-native-fbank + metadata 参数 + 缺参硬拒 + golden test |
| transducer 状态机 ~90 张量易错 | 静默乱码非崩溃 | metadata 通用零初始化 + 对齐 sherpa 输出 |
| 中英流式 CTC 模型可能不存在 | CTC 简化路径受阻 | 默认用 transducer；CTC 作可选 |
| DirectML 动态轴慢 5× / 注意力拷贝税 | 流式延迟 | 一切定形导出；KV 定长 |
| EPContext 缓存因 EP/驱动更新失效 | 启动报错 | 打戳缓存名 + 后台重编 + CPU 兜底 |
| Whisper-genai 不继承 WinML EP 选择 | 跨硬件统一性破 | 第二遍默认 SenseVoice，不用 Whisper；genai 作可选插件 |
| NAudio MSIX Release 裁剪 | 打包崩溃 | 排除 NAudio* 裁剪 + 早测 (若走 MSIX) |
| NPU 量化掉精度 (流式逐 chunk 累积) | WER/CER 上升 | QDQ 校准、per-channel、对齐 float 基线 |

---

## 18. 待解决决策

1. **§13 打包形态** — 确认 unpackaged 自包含 (默认推荐) vs MSIX。
2. **目标语言广度** — v1 聚焦 zh-en；是否需把 SenseVoice 的 ja/ko/yue 或 Whisper(99 语言, 经 genai 插件) 纳入 UI 语言选择。
3. **流式首遍备选** — 是否同时支持流式 CTC (取决于可用的中英流式 CTC 模型质量)。

---

## 附录：关键来源

- Windows ML: [overview](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/overview) · [supported EPs](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/supported-execution-providers) · [select EPs](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/select-execution-providers) · [distribute](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/distributing-your-app)
- ONNX Runtime: [C# OrtValue/IO binding](https://onnxruntime.ai/docs/api/csharp/api/Microsoft.ML.OnnxRuntime.OrtValue.html) · [threading](https://onnxruntime.ai/docs/performance/tune-performance/threading.html) · [EPContext](https://onnxruntime.ai/docs/execution-providers/EP-Context-Design.html) · [QNN EP](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html)
- sherpa-onnx 参考: [online-zipformer2-transducer-model.cc](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/online-zipformer2-transducer-model.cc) · [features.h](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/features.h) · icefall [export-onnx-streaming.py](https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/zipformer/export-onnx-streaming.py)
- 特征: [kaldi-native-fbank](https://github.com/csukuangfj/kaldi-native-fbank) · [FunASR wav_frontend](https://github.com/modelscope/FunASR/blob/main/funasr/frontends/wav_frontend.py) · [whisper audio.py](https://github.com/openai/whisper/blob/main/whisper/audio.py)
- WinUI3: [MVVM Toolkit](https://learn.microsoft.com/en-us/dotnet/communitytoolkit/mvvm) · [DI tutorial](https://learn.microsoft.com/en-us/windows/apps/tutorials/winui-mvvm-toolkit/dependency-injection) · [Channels](https://learn.microsoft.com/en-us/dotnet/core/extensions/channels) · [DispatcherQueue.TryEnqueue](https://learn.microsoft.com/en-us/windows/windows-app-sdk/api/winrt/microsoft.ui.dispatching.dispatcherqueue.tryenqueue) · [file access](https://learn.microsoft.com/en-us/windows/apps/develop/files/file-access-permissions)
- 音频: [NAudio WasapiRecorder](https://github.com/naudio/NAudio/blob/main/Docs/WasapiRecorder.md) · [AudioGraph](https://learn.microsoft.com/en-us/windows/apps/develop/media-authoring-processing/audio-graphs)
- genai: [onnxruntime-genai](https://github.com/microsoft/onnxruntime-genai) · [WinML genai](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/run-genai-onnx-models)
- Silero VAD: [repo](https://github.com/snakers4/silero-vad)
