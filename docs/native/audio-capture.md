# Audio capture (WASAPI)

`Stt.Audio.Windows.WasapiAudioCapture` is the real microphone source (spec §4). It is the only
OS-bound audio dependency and is isolated in its own `net10.0-windows` project so `Stt.Core` stays
headless-testable (Core tests use `FileAudioCapture` instead).

## Behavior

- Opens the default capture endpoint (`new WasapiCapture()`) or a selected `MMDevice`.
- Converts the device mix format (commonly 32-bit float, 44.1/48 kHz, stereo) to **16 kHz mono
  float** via `Resampler.ToMono16k`, then emits `AudioFrame`s of `frameSamples` (default 512 = one
  Silero VAD window).
- Microphone-denied surfaces as `UnauthorizedAccessException` from `StartAsync` — the app catches
  it, prompts, and stops recording (spec §12, §14).

## Manual smoke test (cannot run in CI — needs a device)

```csharp
using var cap = new WasapiAudioCapture();           // default mic
int frames = 0;
cap.FrameAvailable += f => { frames += 1; };
var cts = new CancellationTokenSource(TimeSpan.FromSeconds(3));
try { await cap.StartAsync(cts.Token); } catch (OperationCanceledException) { }
Console.WriteLine($"captured {frames} frames (~{frames * 512 / 16000.0:F1}s)");
```

Expected: roughly `3 s × 16000 / 512 ≈ 93` frames.

## Known limitation (Phase 1 refinement)

`Resampler.ToMono16k` is stateless, so resampling each `DataAvailable` buffer independently
introduces a tiny discontinuity at buffer boundaries. For non-integer source rates (44.1 kHz) a
stateful streaming resampler that carries filter history across buffers is a Phase 1 improvement;
integer rates (48 kHz → 16 kHz) and the offline file path are unaffected in practice.

## Device enumeration

`WasapiAudioCapture.EnumerateCaptureDevices()` returns active capture endpoints for the UI device
picker.
