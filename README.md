# BPM Timecoded Buffer (VJ.Tools)

Daydream Scope plugin that stamps beat-grid timecodes on video frames using Ableton Link and preserves them through AI generation via VACE masking.

The barcode survives diffusion — decode it on AI output to know the exact beat position that produced each frame, regardless of variable inference latency.

## How it works

```
Input Video  -->  [Stamp Barcode]  -->  [VACE Mask]  -->  Scope AI  -->  [Decode & Buffer]  -->  Output
                       |                    |                              |
                  Link beat clock      mask=0: preserve            Barcode survives!
                  encodes beat#        mask=1: AI generates        Beat-synced buffer
                  into bottom 16px     (barcode protected)         modes for playback
```

### Preprocessor (BPM Timecoded Buffer)
1. **Ableton Link** provides a shared beat clock (joins any Link session on the network)
2. **BCH barcode** encodes beat position, BPM, frame sequence into a 16px strip at the bottom of each frame
3. **VACE mask** tells Scope's diffusion pipeline to preserve the barcode (mask=0) while generating content above it (mask=1)
4. **ControlNet** (optional) extracts canny/depth/scribble from the content region to guide AI generation

### Postprocessor (BPM Timecode Buffer Output)
1. **Decodes barcode** from AI output to recover beat position, BPM, and frame sequence
2. **Strips barcode** from output so viewers don't see it
3. **Buffer modes** for beat-accurate playback:
   - **Strip** — pass-through, just remove the barcode
   - **Beat-quantized** — hold frames until the next beat boundary, release the closest frame to each beat
   - **Loop** — record N beats of frames, then loop playback beat-synced
   - **Latency** — FIFO buffer with configurable delay to smooth out inference jitter
4. **Ableton Link sync** — postprocessor can join a Link session for accurate local beat timing

## Ableton Link

Both the preprocessor and postprocessor support Ableton Link for beat synchronization.

- **Preprocessor**: Always uses Link when `aalink` is installed. Stamps frames with the shared beat clock so any Link peer (Ableton Live, VCV Rack, etc.) stays in sync.
- **Postprocessor**: Toggle **Ableton Link Sync** to join a Link session. When enabled, beat-quantized and loop modes use Link's beat position for timing instead of relying solely on decoded barcode beats. This is especially useful when running Scope locally.
- **Fallback**: If `aalink` is not installed (e.g., cloud/RunPod), a free-running internal clock is used. The postprocessor can still use decoded barcode beats for timing.

```bash
# Install with Ableton Link support (recommended for local use)
pip install ".[link]"
```

## MIDI Mapping

All runtime parameters are automatically MIDI-mappable in Scope. Any Pydantic field with `ui_field_config` can be mapped to a MIDI controller — BPM, buffer mode, loop length, beat hold window, latency, and more.

## Install

```bash
# Basic install
pip install .

# With Ableton Link support (recommended for local use)
pip install ".[link]"

# On RunPod / cloud (no Link needed, barcode timing is sufficient)
uv pip install --no-deps git+https://github.com/VJ-Tools/bpm-timecoded-buffer.git
```

Scope discovers the plugin automatically via the `scope` entry point.

## Parameters

### Preprocessor

| Parameter | Default | Range | MIDI | Description |
|-----------|---------|-------|------|-------------|
| `initial_bpm` | 120.0 | 60-300 | — | Starting BPM (load-time) |
| `barcode_height` | 16 | 4-128 | Yes | Barcode strip height in pixels |
| `control_mode` | none | dropdown | Yes | ControlNet: none, canny, depth, scribble |
| `canny_low` | 50 | 0-255 | Yes | Canny edge low threshold |
| `canny_high` | 150 | 0-255 | Yes | Canny edge high threshold |
| `mask_feather` | 2 | 0-16 | Yes | Soft transition at barcode boundary (px) |
| `strip_barcode` | false | toggle | Yes | Black out barcode in display output |
| `test_input` | false | toggle | Yes | Use test pattern instead of video input |
| `tap_tempo` | false | toggle | Yes | Tap to detect BPM (for no-Link setups) |

### Postprocessor

| Parameter | Default | Range | MIDI | Description |
|-----------|---------|-------|------|-------------|
| `barcode_height` | 16 | 4-128 | Yes | Barcode strip height in pixels |
| `buffer_mode` | strip | dropdown | Yes | Buffer mode: strip, beat, loop, latency |
| `loop_length_beats` | 8 | 1-64 | Yes | Loop recording length in beats |
| `latency_delay_ms` | 100 | 0-2000 | Yes | Latency buffer delay in milliseconds |
| `beat_hold_beats` | 2 | 1-64 | Yes | Beat-quantized hold window |
| `link_sync` | false | toggle | Yes | Enable Ableton Link sync |
| `link_bpm` | 120.0 | 20-999 | Yes | Link BPM (read-only when synced) |
| `loop_reset` | false | toggle | Yes | Reset loop recording |
| `latency_nudge_ms` | 10 | 1-100 | Yes | Latency nudge step size |

## Barcode Spec

- **BCH(71,50,3)**: corrects up to 3 bit errors per frame
- **85 bars**: 14 sync + 71 data/ECC, 6px per bar, 16px tall
- **ITU-R limited range**: Y=16 (black), Y=235 (white)
- **50-bit payload**:
  - `beatWhole` (12b): beat number 0-4095
  - `beatFrac` (8b): fractional beat 0-255
  - `frameSeq` (14b): frame sequence 0-16383
  - `bpm` (9b): BPM 60-571 (offset encoding)
  - `flags` (7b): reserved

## Test

```bash
pip install ".[dev]"
pytest tests/ -v
```

13 tests covering barcode encode/decode roundtrip, VACE masking, control modes, buffer modes, tap tempo, and postprocessor pipeline.

## License

MIT — Part of [OpticMystic's VJ.Tools](https://github.com/VJ-Tools) collection
