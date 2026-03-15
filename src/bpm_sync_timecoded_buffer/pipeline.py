"""
BPM Timecoded Buffer — Beat-Grid Timecoding Pipeline for Daydream Scope

Preprocessor that:
1. Joins an Ableton Link session, listens for MIDI clock, or receives OSC beat data
2. Stamps a VJSync barcode on each input frame (encoding current beat position)
3. Creates a VACE inpainting mask at generation resolution to preserve the barcode

The barcode is protected through two layers:
  - VACE masking: mask=0 at barcode strip tells the AI to preserve that region
  - BCH(71,50,3) error correction: corrects up to 3 bit errors as backup

The VACE mask is generated at the main pipeline's GENERATION resolution
(height/width/input_size from kwargs), NOT the input video resolution.
Scope broadcasts all pipeline parameters to every pipeline in the chain,
so the preprocessor reads the generation dimensions from kwargs.

Clock sources:
  - Ableton Link: Networked beat sync with DAWs and other Link-enabled apps
  - MIDI Clock: Standard MIDI timing (24 PPQN) from DJ software, drum machines, etc.
  - OSC: Beat position and BPM pushed from external software via /scope/osc_beat
  - Internal: Free-running clock at configured BPM (fallback when no external sync)

Barcode spec:
  - 16px tall, full frame width, bottom of frame
  - BCH(71,50,3): corrects up to 3 bit errors
  - Payload: beatWhole(12b) + beatFrac(8b) + frameSeq(14b) + bpm(9b) + flags(7b)
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import torch

from .vjsync_codec import (
    VJSyncPayload,
    STRIP_HEIGHT,
    encode_bpm,
    encode_beat_frac,
    decode_bpm,
    decode_beat_frac,
    stamp_barcode,
    read_barcode,
)
from .test_source import TestPatternSource
from .midi_clock import MidiClock

# --- Scope imports: match actual Scope package structure ---
try:
    from scope.core.pipelines.interface import Pipeline, Requirements
    from scope.core.pipelines.base_schema import (
        BasePipelineConfig, UsageType, ModeDefaults, ui_field_config,
    )
    _HAS_SCOPE = True
except ImportError:
    # Fallback for running outside Scope (standalone tests)
    class Pipeline:
        pass
    class Requirements:
        def __init__(self, input_size: int = 1):
            self.input_size = input_size
    class BasePipelineConfig:
        pass
    class UsageType:
        PREPROCESSOR = "preprocessor"
        POSTPROCESSOR = "postprocessor"
    class ModeDefaults:
        def __init__(self, default=False):
            self.default = default
    def ui_field_config(**kwargs):
        return kwargs
    _HAS_SCOPE = False

# Also try importing Field from pydantic (only needed when Scope is present)
try:
    from pydantic import Field
except ImportError:
    def Field(**kwargs):
        return kwargs.get("default")

logger = logging.getLogger(__name__)


# --- Clock Source Enum ---

class ClockSource(str, Enum):
    LINK = "link"          # Ableton Link (networked beat sync)
    MIDI_CLOCK = "midi_clock"  # MIDI clock (24 PPQN from external device)
    OSC = "osc"            # OSC-driven clock (BPM + beat pushed via /scope/osc_beat)
    INTERNAL = "internal"  # Free-running internal clock


# --- Ableton Link wrapper (runs in background thread) ---

class LinkClock:
    """
    Thin wrapper around aalink that runs the async event loop in a background
    thread. Provides synchronous getters for beat/tempo/phase.
    """

    def __init__(self, initial_bpm: float = 120.0):
        self._beat = 0.0
        self._tempo = initial_bpm
        self._phase = 0.0
        self._num_peers = 0
        self._enabled = False
        self._link = None
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop_event = threading.Event()

    def start(self, bpm: float = 120.0):
        """Start the Link session in a background thread."""
        if self._thread is not None:
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop, args=(bpm,), daemon=True, name="link-clock"
        )
        self._thread.start()
        logger.info(f"[BPM Buffer/Link] Clock started at {bpm} BPM")

    def stop(self):
        """Stop the Link session."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._enabled = False
        logger.info("[BPM Buffer/Link] Clock stopped")

    def _run_loop(self, bpm: float):
        """Background thread: run async Link polling loop."""
        try:
            from aalink import Link
        except ImportError:
            logger.warning(
                "[BPM Buffer/Link] aalink not installed -- using free-running clock. "
                "Install with: pip install aalink"
            )
            self._run_freerunning(bpm)
            return

        loop = asyncio.new_event_loop()
        self._loop = loop

        async def poll():
            link = Link(bpm)
            link.enabled = True
            self._link = link
            self._enabled = True
            self._tempo = bpm

            logger.info(f"[BPM Buffer/Link] Ableton Link session joined at {bpm} BPM")

            while not self._stop_event.is_set():
                try:
                    beat_val = await asyncio.wait_for(
                        link.sync(1 / 16), timeout=0.1
                    )
                    self._beat = beat_val
                    self._tempo = bpm
                    self._phase = beat_val % 4.0
                except asyncio.TimeoutError:
                    pass
                except Exception as e:
                    if not self._stop_event.is_set():
                        logger.debug(f"[BPM Buffer/Link] poll error: {e}")
                    await asyncio.sleep(0.016)

            link.enabled = False
            self._enabled = False

        try:
            loop.run_until_complete(poll())
        except Exception as e:
            logger.error(f"[BPM Buffer/Link] Loop error: {e}")
        finally:
            loop.close()

    def _run_freerunning(self, bpm: float):
        """Fallback: free-running beat clock when aalink is not available."""
        self._enabled = True
        self._tempo = bpm
        start_time = time.monotonic()

        while not self._stop_event.is_set():
            elapsed = time.monotonic() - start_time
            beats = elapsed * (bpm / 60.0)
            self._beat = beats
            self._phase = beats % 4.0
            time.sleep(0.008)  # ~125 Hz

        self._enabled = False

    @property
    def beat(self) -> float:
        return self._beat

    @property
    def tempo(self) -> float:
        return self._tempo

    @property
    def phase(self) -> float:
        return self._phase

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def num_peers(self) -> int:
        return self._num_peers


class BufferMode(str, Enum):
    NO_BUFFER = "no_buffer"  # Just strip barcode, pass through (simplest)
    LATENCY = "latency"      # Adjustable latency: wall-clock FIFO, 0-60s delay slider
    BEAT = "beat"            # Beat-delayed: smooth full-framerate playback, N beats behind


# --- Unified Clock Manager ---

class ClockManager:
    """
    Manages multiple clock sources (Link, MIDI Clock, OSC, Internal) and provides
    a unified beat/tempo/phase interface. Only one source active at a time.

    OSC mode: beat and BPM are pushed from external software via Scope's OSC
    parameter mapping. External app sends:
      /scope/osc_beat <float>    — current beat position (continuously updated)
      /scope/clock_bpm <float>   — current BPM
    The pipeline reads these from kwargs on each __call__.
    """

    def __init__(self, initial_bpm: float = 120.0):
        self._source = ClockSource.INTERNAL
        self._link_clock: Optional[LinkClock] = None
        self._midi_clock: Optional[MidiClock] = None
        self._internal_bpm = initial_bpm
        self._internal_beat = 0.0
        self._internal_start = time.monotonic()
        self._midi_device = ""
        # OSC-driven state (updated from kwargs each frame)
        self._osc_beat: float = 0.0
        self._osc_bpm: float = initial_bpm

    def set_source(self, source: ClockSource, bpm: float = 120.0, midi_device: str = ""):
        """Switch clock source. Stops previous source, starts new one."""
        if source == self._source and midi_device == self._midi_device:
            # Same source — just update BPM for internal/OSC
            if source == ClockSource.INTERNAL:
                self._internal_bpm = bpm
            elif source == ClockSource.OSC:
                self._osc_bpm = bpm
            return

        # Stop current source
        self._stop_current()

        self._source = source
        self._midi_device = midi_device

        if source == ClockSource.LINK:
            self._link_clock = LinkClock(bpm)
            self._link_clock.start(bpm)
            logger.info(f"[Clock] Switched to Ableton Link at {bpm} BPM")

        elif source == ClockSource.MIDI_CLOCK:
            self._midi_clock = MidiClock()
            self._midi_clock.start(device_name=midi_device)
            logger.info(f"[Clock] Switched to MIDI Clock on '{midi_device or 'default'}'")

        elif source == ClockSource.OSC:
            self._osc_bpm = bpm
            self._osc_beat = 0.0
            logger.info(f"[Clock] Switched to OSC clock (send /scope/osc_beat and /scope/clock_bpm)")

        else:  # ClockSource.INTERNAL
            self._internal_bpm = bpm
            self._internal_start = time.monotonic()
            self._internal_beat = 0.0
            logger.info(f"[Clock] Switched to Internal clock at {bpm} BPM")

    def update_osc(self, beat: float, bpm: float):
        """Update OSC-driven beat and BPM (called from pipeline __call__ each frame)."""
        self._osc_beat = beat
        if bpm > 0:
            self._osc_bpm = bpm

    def _stop_current(self):
        """Stop the currently active clock source."""
        if self._link_clock is not None:
            self._link_clock.stop()
            self._link_clock = None
        if self._midi_clock is not None:
            self._midi_clock.stop()
            self._midi_clock = None

    def stop(self):
        """Stop all clock sources."""
        self._stop_current()

    def set_internal_bpm(self, bpm: float):
        """Update internal clock BPM."""
        self._internal_bpm = max(20.0, min(999.0, bpm))

    @property
    def beat(self) -> float:
        if self._source == ClockSource.LINK and self._link_clock:
            return self._link_clock.beat
        elif self._source == ClockSource.MIDI_CLOCK and self._midi_clock:
            return self._midi_clock.beat
        elif self._source == ClockSource.OSC:
            return self._osc_beat
        else:
            # Internal free-running
            elapsed = time.monotonic() - self._internal_start
            return elapsed * (self._internal_bpm / 60.0)

    @property
    def tempo(self) -> float:
        if self._source == ClockSource.LINK and self._link_clock:
            return self._link_clock.tempo
        elif self._source == ClockSource.MIDI_CLOCK and self._midi_clock:
            t = self._midi_clock.tempo
            return t if t > 0 else self._internal_bpm  # Fallback if no clock received yet
        elif self._source == ClockSource.OSC:
            return self._osc_bpm
        else:
            return self._internal_bpm

    @property
    def phase(self) -> float:
        return self.beat % 4.0

    @property
    def enabled(self) -> bool:
        if self._source == ClockSource.LINK and self._link_clock:
            return self._link_clock.enabled
        elif self._source == ClockSource.MIDI_CLOCK and self._midi_clock:
            return self._midi_clock.enabled
        return True  # Internal is always enabled

    @property
    def source(self) -> ClockSource:
        return self._source

    @property
    def source_info(self) -> dict:
        """Return diagnostic info about the current clock source."""
        info = {"source": self._source.value, "beat": self.beat, "tempo": self.tempo}
        if self._source == ClockSource.LINK and self._link_clock:
            info["link_peers"] = self._link_clock.num_peers
            info["link_enabled"] = self._link_clock.enabled
        elif self._source == ClockSource.MIDI_CLOCK and self._midi_clock:
            info["midi_device"] = self._midi_clock.device_name
            info["midi_running"] = self._midi_clock.running
            info["midi_enabled"] = self._midi_clock.enabled
        elif self._source == ClockSource.OSC:
            info["osc_beat"] = self._osc_beat
            info["osc_bpm"] = self._osc_bpm
        return info


# --- Config ---

if _HAS_SCOPE:
    class BpmBufferConfig(BasePipelineConfig):
        """Configuration schema for BPM Timecoded Buffer preprocessor."""

        # --- Class attributes (no type annotation = plain class var, not Pydantic field) ---
        pipeline_id = "bpm_sync_timecoded_buffer__vj_tools"
        pipeline_name = "BPM Sync Timecoded Buffer (VJ.Tools)"
        pipeline_description = (
            "Beat-grid timecoding via Ableton Link or MIDI Clock. Stamps VJSync "
            "barcode on input, preserves through AI via VACE masking (mask=0 at "
            "barcode). Client decodes surviving barcode on output for beat-accurate "
            "timing."
        )
        supports_prompts = False
        modified = True
        usage = [UsageType.PREPROCESSOR]
        modes = {
            "video": ModeDefaults(default=True),
            "text": ModeDefaults(default=True),
        }

        # --- Runtime parameters ---

        clock_source: ClockSource = Field(
            default=ClockSource.LINK,
            json_schema_extra=ui_field_config(
                order=0,
                label="Clock Source",
                is_load_param=False,
            ),
        )

        clock_bpm: float = Field(
            default=120.0,
            ge=20.0,
            le=999.0,
            json_schema_extra=ui_field_config(
                order=1,
                label="Clock BPM (Link/Internal)",
                is_load_param=False,
            ),
        )

        midi_device: str = Field(
            default="",
            json_schema_extra=ui_field_config(
                order=2,
                label="MIDI Clock Device",
                is_load_param=False,
            ),
        )

        osc_beat: float = Field(
            default=0.0,
            ge=0.0,
            json_schema_extra=ui_field_config(
                order=3,
                label="OSC Beat Position",
                category="input",
            ),
        )

        test_input: bool = Field(
            default=False,
            json_schema_extra=ui_field_config(
                order=5,
                label="Test Pattern Input",
                is_load_param=False,
            ),
        )
else:
    class BpmBufferConfig:
        """Standalone config (no Pydantic) for testing outside Scope."""
        def __init__(self, **kwargs):
            self.pipeline_id = kwargs.get("pipeline_id", "bpm_timecoded_buffer")
            self.pipeline_name = kwargs.get("pipeline_name", "BPM Sync Timecoded Buffer (VJ.Tools)")
            self.clock_source = kwargs.get("clock_source", "link")
            self.clock_bpm = kwargs.get("clock_bpm", 120.0)
            self.midi_device = kwargs.get("midi_device", "")
            self.osc_beat = kwargs.get("osc_beat", 0.0)
            self.test_input = kwargs.get("test_input", False)


# --- Pipeline ---

class BpmTimecodedBufferPipeline(Pipeline):
    """
    Scope preprocessor that stamps beat-grid timecodes using Ableton Link
    or MIDI Clock and creates VACE masks to preserve them through AI generation.

    Clock sources:
      - Ableton Link: Networked beat sync (default)
      - MIDI Clock: 24 PPQN from DJ software, drum machines, hardware sequencers
      - Internal: Free-running at configured BPM (fallback)

    The barcode is stamped on input, the VACE mask ensures AI generates everything
    EXCEPT the barcode strip (mask=0 = preserve), and the client decodes the
    surviving barcode on the output for beat-accurate timing.

    VACE mask format (matching Scope's VaceEncodingBlock):
      - vace_input_masks [1,1,F,H,W] binary: 1=generate, 0=preserve
    """

    @classmethod
    def get_config_class(cls):
        return BpmBufferConfig

    def __init__(
        self,
        config=None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float16,
        **kwargs,  # Scope passes height, width, quantization, loras, etc.
    ):
        if config is None:
            config = BpmBufferConfig() if _HAS_SCOPE else type('Config', (), kwargs)()
        self.config = config
        self.device = (
            device if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = dtype if self.device.type == "cuda" else torch.float32
        self._frame_seq = 0

        # Initialize clock manager with configured source
        initial_bpm = getattr(config, "clock_bpm", 120.0)
        self._clock = ClockManager(initial_bpm)

        source = ClockSource(getattr(config, "clock_source", "link"))
        midi_device = getattr(config, "midi_device", "")
        self._clock.set_source(source, bpm=initial_bpm, midi_device=midi_device)

        # Test pattern source (lazy-initialized on first use)
        self._test_source: Optional[TestPatternSource] = None

        logger.info(
            f"[BPM Buffer] Pipeline initialized "
            f"(device={self.device}, clock={source.value})"
        )

    def __del__(self):
        if hasattr(self, "_clock"):
            self._clock.stop()

    def set_bpm(self, bpm: float):
        """Manually set BPM (for RunPod / no-Link scenarios)."""
        bpm = max(20.0, min(999.0, bpm))
        self._clock.set_internal_bpm(bpm)
        if self._test_source:
            self._test_source.set_bpm(bpm)
        logger.info(f"[BPM Buffer] Manual BPM: {bpm:.1f}")

    def prepare(self, **kwargs) -> "Requirements":
        """Tell Scope how many input frames we need per call."""
        return Requirements(input_size=1)

    # VAE alignment — diffusion models require spatial dims divisible by this
    _VAE_ALIGN = 8

    @staticmethod
    def _align(value: int, alignment: int) -> int:
        """Round up to nearest multiple of alignment."""
        return ((value + alignment - 1) // alignment) * alignment

    def __call__(self, **kwargs) -> dict:
        """
        Process input frames (Scope Pipeline interface).

        Scope passes video as kwargs["video"] — a list of (1, H, W, C) uint8 tensors.

        Returns:
            dict with video, vace_input_masks
        """
        video = kwargs.get("video", [])
        # Barcode height is fixed by VJSync codec spec — not configurable.
        # The decoder auto-detects via sync pattern, so encoder must match exactly.
        barcode_h = STRIP_HEIGHT
        test_input = kwargs.get("test_input", getattr(self.config, "test_input", False))

        # --- Clock source switching (runtime parameter updates) ---
        clock_source_str = kwargs.get("clock_source", getattr(self.config, "clock_source", "link"))
        clock_bpm = kwargs.get("clock_bpm", getattr(self.config, "clock_bpm", 120.0))
        midi_device = kwargs.get("midi_device", getattr(self.config, "midi_device", ""))

        # Convert string to enum safely
        try:
            target_source = ClockSource(str(clock_source_str))
        except ValueError:
            target_source = ClockSource.LINK

        # Switch clock source if changed
        if target_source != self._clock.source:
            self._clock.set_source(target_source, bpm=clock_bpm, midi_device=midi_device)
        elif target_source == ClockSource.INTERNAL:
            self._clock.set_internal_bpm(clock_bpm)

        # Update OSC clock from parameters (pushed via /scope/osc_beat and /scope/clock_bpm)
        if target_source == ClockSource.OSC:
            osc_beat = kwargs.get("osc_beat", getattr(self.config, "osc_beat", 0.0))
            self._clock.update_osc(beat=osc_beat, bpm=clock_bpm)

        # Handle both list and tensor input, or empty
        if isinstance(video, list) and len(video) == 0:
            return {"video": torch.zeros(1, 1, 1, 3)}

        # --- Test pattern override ---
        if test_input:
            ref = video[0]
            _, H, W, _ = ref.shape
            if self._test_source is None or self._test_source.width != W or self._test_source.height != H:
                self._test_source = TestPatternSource(width=W, height=H)
            video = self._test_source.generate_batch(
                self._clock, num_frames=len(video), barcode_height=STRIP_HEIGHT
            )

        # Stack frames
        frames = torch.cat(video, dim=0).float()  # (F, H, W, C), [0, 255]
        F, H, W, C = frames.shape

        # --- 1. Stamp barcode on each frame using beat clock ---
        frames_np = frames.cpu().numpy().astype(np.uint8)

        for f_idx in range(F):
            beat = self._clock.beat
            bpm = self._clock.tempo

            beat_whole = int(beat) & 0xFFF
            beat_frac = encode_beat_frac(beat - int(beat))
            bpm_enc = encode_bpm(bpm)

            payload = VJSyncPayload(
                beat_whole=beat_whole,
                beat_frac=beat_frac,
                frame_seq=self._frame_seq & 0x3FFF,
                bpm_encoded=bpm_enc,
                flags=0,
            )

            stamp_barcode(frames_np[f_idx], payload)
            self._frame_seq += 1

        # Convert back to tensor with stamped barcodes
        frames_stamped = torch.from_numpy(frames_np).float()

        # --- 2. Video output ---
        frames_01 = frames_stamped / 255.0  # (F, H, W, C), [0, 1]

        result = {"video": frames_01}

        # --- 3. Generate VACE mask at generation resolution ---
        # Scope broadcasts pipeline params to every pipeline in the chain.
        # We need height, width, AND frame count to match the main pipeline's
        # VaceEncodingBlock expectations.
        #
        # Log all kwargs on first few calls so we can diagnose param names.
        if self._frame_seq <= F * 3:
            kwarg_keys = {k: type(v).__name__ for k, v in kwargs.items() if k != "video"}
            logger.info(f"[BPM Buffer] kwargs keys: {kwarg_keys}")

        gen_h = int(kwargs.get("height", self._align(H, self._VAE_ALIGN)))
        gen_w = int(kwargs.get("width", self._align(W, self._VAE_ALIGN)))

        # Frame count: try multiple kwarg names that Scope might use.
        # The main pipeline processes chunks of N frames (e.g. 13), but the
        # preprocessor only sees 1 frame at a time. We need the main pipeline's
        # frame count to size the VACE mask correctly.
        gen_frames = None
        for key in ("input_size", "num_frames", "video_length", "chunk_size", "num_inference_steps"):
            val = kwargs.get(key)
            if val is not None:
                gen_frames = int(val)
                if self._frame_seq <= F * 3:
                    logger.info(f"[BPM Buffer] Found frame count from '{key}' = {gen_frames}")
                break

        if gen_frames is None or gen_frames < 1:
            # Cannot determine main pipeline's frame count — skip VACE mask
            # to avoid shape mismatch. Barcode still survives via BCH error
            # correction (corrects up to 3 bit errors per codeword).
            if self._frame_seq <= F * 3:
                logger.warning(
                    f"[BPM Buffer] Cannot determine generation frame count from kwargs. "
                    f"Skipping vace_input_masks. Barcode relies on BCH error correction only."
                )
        else:
            # Ensure alignment even if Scope passes non-aligned values
            gen_h = self._align(gen_h, self._VAE_ALIGN)
            gen_w = self._align(gen_w, self._VAE_ALIGN)

            # Scale barcode height proportionally to generation resolution
            scale_y = gen_h / H if H > 0 else 1.0
            barcode_h_gen = max(4, round(barcode_h * scale_y))

            # mask=1 -> AI generates (content area)
            # mask=0 -> preserve (barcode strip at bottom)
            vace_mask = torch.ones(gen_frames, gen_h, gen_w, dtype=torch.float32)
            vace_mask[:, -barcode_h_gen:, :] = 0.0  # Preserve barcode strip

            # Reshape to [B=1, C=1, F, H, W] as expected by VaceEncodingBlock
            vace_mask = vace_mask.unsqueeze(0).unsqueeze(0)
            vace_mask = vace_mask.to(device=self.device, dtype=self.dtype)

            result["vace_input_masks"] = vace_mask

            if self._frame_seq <= F * 3:
                logger.info(
                    f"[BPM Buffer] Input {F}×{H}×{W} → VACE mask "
                    f"[1,1,{gen_frames},{gen_h},{gen_w}], "
                    f"barcode={barcode_h}px→{barcode_h_gen}px"
                )

        # Clock state for diagnostics
        result["_bpm_buffer_meta"] = self._clock.source_info
        result["_bpm_buffer_meta"]["frame_seq"] = self._frame_seq

        return result


# --- Postprocessor Config ---

if _HAS_SCOPE:
    class BpmStripConfig(BasePipelineConfig):
        """Configuration schema for BPM Timecoded Buffer postprocessor."""

        # Class attributes (no annotation = not a Pydantic field)
        pipeline_id = "bpm_sync_timecoded_buffer_output__vj_tools"
        pipeline_name = "BPM Sync Timecoded Buffer Output (VJ.Tools)"
        pipeline_description = (
            "Postprocessor that decodes timecode barcodes from AI output and "
            "provides latency-compensated or beat-quantized buffering. "
            "Strips the barcode from output so viewers don't see it."
        )
        supports_prompts = False
        modified = True
        usage = [UsageType.POSTPROCESSOR]
        modes = {
            "video": ModeDefaults(default=True),
            "text": ModeDefaults(default=True),
        }

        # Pydantic fields — performance controls in Input & Controls (category="input")
        # for MIDI mapping via Scope's native MIDI support

        buffer_mode: BufferMode = Field(
            default=BufferMode.LATENCY,
            json_schema_extra=ui_field_config(
                order=0,
                label="Buffer Mode",
                category="input",
            ),
        )

        latency_delay_ms: int = Field(
            default=100,
            ge=0,
            le=60000,
            json_schema_extra=ui_field_config(
                order=1,
                label="Latency Buffer (ms)",
                category="input",
            ),
        )

        beat_buffer_depth: int = Field(
            default=4,
            ge=1,
            le=64,
            json_schema_extra=ui_field_config(
                order=2,
                label="Beat Buffer Depth",
                category="input",
            ),
        )

        hold: bool = Field(
            default=False,
            json_schema_extra=ui_field_config(
                order=3,
                label="HOLD (freeze playback)",
                category="input",
            ),
        )

        reset_buffer: bool = Field(
            default=False,
            json_schema_extra=ui_field_config(
                order=4,
                label="Reset Buffer",
                category="input",
            ),
        )

        # Clock source — configuration panel (not performance controls)

        clock_source: ClockSource = Field(
            default=ClockSource.LINK,
            json_schema_extra=ui_field_config(
                order=5,
                label="Clock Source",
            ),
        )

        clock_bpm: float = Field(
            default=120.0,
            ge=20.0,
            le=999.0,
            json_schema_extra=ui_field_config(
                order=6,
                label="Clock BPM (Link/Internal)",
            ),
        )

        midi_device: str = Field(
            default="",
            json_schema_extra=ui_field_config(
                order=7,
                label="MIDI Clock Device",
            ),
        )

        osc_beat: float = Field(
            default=0.0,
            ge=0.0,
            json_schema_extra=ui_field_config(
                order=8,
                label="OSC Beat Position",
                category="input",
            ),
        )

        buffer_fill_pct: float = Field(
            default=0.0,
            ge=0.0,
            le=100.0,
            json_schema_extra=ui_field_config(
                order=10,
                label="Buffer Fill %",
            ),
        )
else:
    class BpmStripConfig:
        """Standalone config for testing outside Scope."""
        def __init__(self, **kwargs):
            self.pipeline_id = kwargs.get("pipeline_id", "bpm_sync_timecoded_buffer_output__vj_tools")
            self.pipeline_name = kwargs.get("pipeline_name", "BPM Sync Timecoded Buffer Output (VJ.Tools)")
            self.buffer_mode = kwargs.get("buffer_mode", "latency")
            self.latency_delay_ms = kwargs.get("latency_delay_ms", 100)
            self.beat_buffer_depth = kwargs.get("beat_buffer_depth", 4)
            self.hold = kwargs.get("hold", False)
            self.reset_buffer = kwargs.get("reset_buffer", False)
            self.clock_source = kwargs.get("clock_source", "link")
            self.clock_bpm = kwargs.get("clock_bpm", 120.0)
            self.midi_device = kwargs.get("midi_device", "")
            self.osc_beat = kwargs.get("osc_beat", 0.0)
            self.buffer_fill_pct = kwargs.get("buffer_fill_pct", 0.0)


# --- Buffered Frame ---

@dataclass
class _BufferedFrame:
    """A decoded frame with its timecode metadata."""
    frame: np.ndarray        # (H, W, C) uint8, barcode stripped
    beat: float              # decoded beat position
    bpm: float               # decoded BPM
    frame_seq: int           # decoded frame sequence number
    timestamp: float         # time.monotonic() when received


# --- Postprocessor Pipeline ---

class BpmTimecodeStripPipeline(Pipeline):
    """
    Scope postprocessor that decodes timecode barcodes from AI output,
    strips the barcode, and buffers frames for smooth full-framerate playback.

    Architecture: Wall-clock FIFO + binary search.
    ALL frames are stored with their wall-clock timestamp (time.monotonic()).
    On each render call, we compute a target_time in the past and binary-search
    the FIFO for the closest frame. This gives smooth full-framerate playback
    at any delay depth — no stepped/stutter beat-locked behavior.

    Clock sources (for beat buffer mode):
      - Ableton Link: Networked beat sync with DAWs
      - MIDI Clock: 24 PPQN from DJ software, drum machines
      - Internal: Free-running at configured BPM

    Buffer modes:
      - no_buffer: Just strip the barcode (pass-through, simplest)
      - latency:   Adjustable latency (default) — FIFO + binary search with
                   configurable delay in ms (0-60000). MIDI fader assignable.
      - beat:      Beat-delayed — delay = beat_buffer_depth × ms_per_beat.
    """

    # 60 seconds at ~30fps = 1800 frames max
    MAX_FIFO_FRAMES = 1800
    # Synthetic fallback BPM when no clock source available
    SYNTHETIC_BPM = 120.0

    @classmethod
    def get_config_class(cls):
        return BpmStripConfig

    def __init__(
        self,
        config=None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float16,
        **kwargs,  # Scope passes height, width, quantization, loras, etc.
    ):
        if config is None:
            config = BpmStripConfig() if _HAS_SCOPE else type('Config', (), kwargs)()
        self.config = config
        self.device = (
            device if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = dtype if self.device.type == "cuda" else torch.float32

        # Initialize clock manager
        initial_bpm = getattr(config, "clock_bpm", 120.0)
        self._clock = ClockManager(initial_bpm)

        source_str = getattr(config, "clock_source", "link")
        try:
            source = ClockSource(str(source_str))
        except ValueError:
            source = ClockSource.LINK
        midi_device = getattr(config, "midi_device", "")
        self._clock.set_source(source, bpm=initial_bpm, midi_device=midi_device)

        # --- Wall-clock FIFO buffers ---
        self._beat_fifo: list[_BufferedFrame] = []
        self._latency_fifo: list[_BufferedFrame] = []

        # Current output frame (held between calls when FIFO is empty)
        self._current_output: Optional[np.ndarray] = None

        # Hold state — freezes target_time at the moment hold was engaged
        self._hold_active: bool = False
        self._hold_target_time: float = 0.0

        # Extra delay accumulated during hold
        self._playback_extra_delay: float = 0.0

        # Stats
        self._decode_success = 0
        self._decode_fail = 0

        logger.info(f"[BPM Buffer Output] Postprocessor initialized (clock={source.value})")

    def __del__(self):
        if hasattr(self, "_clock"):
            self._clock.stop()

    def _get_effective_bpm(self) -> float:
        """Get BPM from clock manager → latest frame barcode → synthetic fallback."""
        bpm = self._clock.tempo
        if bpm and bpm > 0:
            return bpm
        # Try from latest frame in beat FIFO
        if self._beat_fifo:
            latest_bpm = self._beat_fifo[-1].bpm
            if latest_bpm > 0:
                return latest_bpm
        return self.SYNTHETIC_BPM

    @staticmethod
    def _binary_search_closest(
        fifo: list[_BufferedFrame], target_time: float
    ) -> Optional[np.ndarray]:
        """
        Binary search the FIFO for the frame closest to target_time.
        Returns the frame's numpy array, or None if FIFO is empty.
        """
        if not fifo:
            return None

        lo, hi = 0, len(fifo) - 1
        while lo < hi:
            mid = (lo + hi) >> 1
            if fifo[mid].timestamp < target_time:
                lo = mid + 1
            else:
                hi = mid

        best = fifo[lo]
        if lo > 0:
            prev = fifo[lo - 1]
            if abs(prev.timestamp - target_time) < abs(best.timestamp - target_time):
                best = prev

        return best.frame

    @staticmethod
    def _evict_old_frames(
        fifo: list[_BufferedFrame], target_time: float, keep_margin: float = 2.0
    ):
        """
        Evict frames older than target_time - keep_margin seconds.
        Always keeps at least 1 frame.
        """
        cutoff = target_time - keep_margin
        while len(fifo) > 1 and fifo[0].timestamp < cutoff:
            fifo.pop(0)

    def prepare(self, **kwargs) -> "Requirements":
        """Tell Scope how many input frames we need per call."""
        return Requirements(input_size=1)

    def __call__(self, **kwargs) -> dict:
        """
        Decode barcode, strip it, and apply buffer mode.

        Scope passes video as kwargs["video"] — a list of (1, H, W, C) uint8 tensors.
        """
        video = kwargs.get("video", [])

        # Handle both list and tensor input, or empty
        if isinstance(video, list) and len(video) == 0:
            return {"video": torch.zeros(1, 1, 1, 3)}

        # Read params from kwargs first (Scope runtime updates), fall back to config
        barcode_h = STRIP_HEIGHT  # Fixed by codec spec
        mode = str(kwargs.get("buffer_mode", getattr(self.config, "buffer_mode", "latency")))
        latency_ms = kwargs.get("latency_delay_ms", getattr(self.config, "latency_delay_ms", 100))
        beat_depth = kwargs.get("beat_buffer_depth", getattr(self.config, "beat_buffer_depth", 4))
        reset_buffer = kwargs.get("reset_buffer", getattr(self.config, "reset_buffer", False))
        hold = kwargs.get("hold", getattr(self.config, "hold", False))

        # --- Clock source switching (runtime parameter updates) ---
        clock_source_str = kwargs.get("clock_source", getattr(self.config, "clock_source", "link"))
        clock_bpm = kwargs.get("clock_bpm", getattr(self.config, "clock_bpm", 120.0))
        midi_device = kwargs.get("midi_device", getattr(self.config, "midi_device", ""))

        try:
            target_source = ClockSource(str(clock_source_str))
        except ValueError:
            target_source = ClockSource.LINK

        if target_source != self._clock.source:
            self._clock.set_source(target_source, bpm=clock_bpm, midi_device=midi_device)
        elif target_source == ClockSource.INTERNAL:
            self._clock.set_internal_bpm(clock_bpm)

        # Update OSC clock from parameters (pushed via /scope/osc_beat and /scope/clock_bpm)
        if target_source == ClockSource.OSC:
            osc_beat = kwargs.get("osc_beat", getattr(self.config, "osc_beat", 0.0))
            self._clock.update_osc(beat=osc_beat, bpm=clock_bpm)

        # --- Reset buffer trigger ---
        if reset_buffer:
            self._beat_fifo.clear()
            self._latency_fifo.clear()
            self._current_output = None
            self._playback_extra_delay = 0.0
            self._hold_active = False
            logger.info("[BPM Buffer Output] Buffer reset")

        # --- Hold toggle ---
        if hold and not self._hold_active:
            self._hold_active = True
            bpm = self._get_effective_bpm()
            ms_per_beat = 60_000.0 / bpm
            delay_s = (beat_depth * ms_per_beat + self._playback_extra_delay) / 1000.0
            self._hold_target_time = time.monotonic() - delay_s
            logger.info(f"[BPM Buffer Output] HOLD engaged at target_time={self._hold_target_time:.3f}")
        elif not hold and self._hold_active:
            bpm = self._get_effective_bpm()
            ms_per_beat = 60_000.0 / bpm
            nominal_delay_s = (beat_depth * ms_per_beat) / 1000.0
            actual_delay_s = time.monotonic() - self._hold_target_time
            self._playback_extra_delay = (actual_delay_s - nominal_delay_s) * 1000.0
            if self._playback_extra_delay < 0:
                self._playback_extra_delay = 0.0
            self._hold_active = False
            logger.info(
                f"[BPM Buffer Output] HOLD released, extra_delay={self._playback_extra_delay:.0f}ms"
            )

        # Stack frames — handle both list of tensors and single tensor
        if isinstance(video, list):
            frames = torch.cat(video, dim=0).float()
        else:
            frames = video.float() if video.dim() == 4 else video.unsqueeze(0).float()

        # Auto-detect [0,1] vs [0,255] range and normalize to [0,255]
        if frames.max() <= 1.0:
            frames = frames * 255.0

        F, H, W, C = frames.shape
        now = time.monotonic()

        # --- Decode barcodes and build buffered frames ---
        incoming: list[_BufferedFrame] = []
        for f_idx in range(F):
            frame_np = frames[f_idx].cpu().numpy().astype(np.uint8)
            payload = read_barcode(frame_np, barcode_h)

            if payload is not None:
                self._decode_success += 1
                beat = payload.beat_whole + decode_beat_frac(payload.beat_frac)
                bpm = decode_bpm(payload.bpm_encoded)
            else:
                self._decode_fail += 1
                beat = self._clock.beat
                bpm = self._clock.tempo if self._clock.tempo > 0 else self.SYNTHETIC_BPM

            # Strip barcode from output
            frame_np[-barcode_h:, :, :] = 0

            incoming.append(_BufferedFrame(
                frame=frame_np,
                beat=beat,
                bpm=bpm,
                frame_seq=payload.frame_seq if payload else 0,
                timestamp=now,
            ))

        # --- Apply buffer mode ---
        if mode == "latency":
            output_frame = self._process_latency(incoming, latency_ms)
        elif mode == "beat":
            output_frame = self._process_beat(incoming, beat_depth)
        else:
            # no_buffer mode: pass through with barcode stripped
            output_frame = incoming[-1].frame if incoming else np.zeros((H, W, C), dtype=np.uint8)

        # Log every 100 frames for diagnostics
        total = self._decode_success + self._decode_fail
        if total % 100 == 1:
            logger.info(
                f"[BPM Buffer Output] mode={mode}, decode={self._decode_success}/{total}, "
                f"clock={self._clock.source.value}, hold={self._hold_active}, "
                f"beat_fifo={len(self._beat_fifo)}, lat_fifo={len(self._latency_fifo)}"
            )

        # Convert single output frame to tensor
        out_tensor = torch.from_numpy(output_frame).float().unsqueeze(0) / 255.0

        result = {"video": out_tensor}

        # --- Buffer fill percentage ---
        if mode == "latency":
            active_fifo_size = len(self._latency_fifo)
        elif mode == "beat":
            active_fifo_size = len(self._beat_fifo)
        else:
            active_fifo_size = 0
        buffer_fill_pct = min(100.0, (active_fifo_size / self.MAX_FIFO_FRAMES) * 100.0)

        if hasattr(self.config, "buffer_fill_pct"):
            self.config.buffer_fill_pct = buffer_fill_pct

        # Diagnostics metadata
        total = self._decode_success + self._decode_fail
        rate = self._decode_success / total if total > 0 else 0.0
        bpm = self._get_effective_bpm()

        clock_info = self._clock.source_info
        result["_bpm_buffer_output_meta"] = {
            "decode_rate": rate,
            "decode_success": self._decode_success,
            "decode_fail": self._decode_fail,
            "buffer_mode": mode,
            "buffer_fill_pct": buffer_fill_pct,
            "beat_fifo_size": len(self._beat_fifo),
            "latency_fifo_size": len(self._latency_fifo),
            "hold_active": self._hold_active,
            "extra_delay_ms": self._playback_extra_delay,
            "effective_bpm": bpm,
            **clock_info,
        }

        return result

    def _process_beat(
        self,
        incoming: list[_BufferedFrame],
        beat_depth: int,
    ) -> np.ndarray:
        """
        Beat-delayed mode — wall-clock FIFO + binary search.
        Smooth full-framerate playback delayed by exactly N beats.
        """
        self._beat_fifo.extend(incoming)
        while len(self._beat_fifo) > self.MAX_FIFO_FRAMES:
            self._beat_fifo.pop(0)

        if not self._beat_fifo:
            return self._current_output if self._current_output is not None else incoming[-1].frame

        bpm = self._get_effective_bpm()
        ms_per_beat = 60_000.0 / bpm
        delay_ms = beat_depth * ms_per_beat + self._playback_extra_delay
        delay_s = delay_ms / 1000.0

        if self._hold_active and self._hold_target_time > 0:
            target_time = self._hold_target_time
        else:
            target_time = time.monotonic() - delay_s

        self._evict_old_frames(self._beat_fifo, target_time, keep_margin=2.0)

        if not self._beat_fifo:
            return self._current_output if self._current_output is not None else incoming[-1].frame

        frame = self._binary_search_closest(self._beat_fifo, target_time)
        if frame is not None:
            self._current_output = frame

        return self._current_output if self._current_output is not None else incoming[-1].frame

    def _process_latency(
        self, incoming: list[_BufferedFrame], delay_ms: int
    ) -> np.ndarray:
        """
        Latency compensation mode — wall-clock FIFO + binary search.
        Smooth and expressive — perfect for MIDI fader control.
        """
        self._latency_fifo.extend(incoming)
        while len(self._latency_fifo) > self.MAX_FIFO_FRAMES:
            self._latency_fifo.pop(0)

        if not self._latency_fifo:
            return self._current_output if self._current_output is not None else incoming[-1].frame

        delay_s = delay_ms / 1000.0
        target_time = time.monotonic() - delay_s

        self._evict_old_frames(self._latency_fifo, target_time, keep_margin=2.0)

        if not self._latency_fifo:
            return self._current_output if self._current_output is not None else incoming[-1].frame

        frame = self._binary_search_closest(self._latency_fifo, target_time)
        if frame is not None:
            self._current_output = frame

        return self._current_output if self._current_output is not None else incoming[-1].frame
