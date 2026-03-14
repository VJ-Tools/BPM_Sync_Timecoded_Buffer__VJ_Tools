"""
BPM Timecoded Buffer — Beat-Grid Timecoding Pipeline for Daydream Scope

Preprocessor that:
1. Joins an Ableton Link session for a shared beat clock
2. Stamps a VJSync barcode on each input frame (encoding current beat position)
3. Creates a VACE inpainting mask that preserves the barcode through AI generation

The barcode survives AI processing via VACE masking (mask=0 = preserve).
The client decodes the surviving barcode on the output to know exactly which
beat produced each frame — regardless of variable Scope inference latency.

VACE integration follows Scope's dual-stream encoding:
  - vace_input_frames: [1, 3, F, H, W] in [-1, 1] — full stamped frames
  - vace_input_masks:  [1, 1, F, H, W] binary — 1=inpaint, 0=preserve
  Scope internally splits: inactive = frame*(1-mask), reactive = frame*mask

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


# --- Config ---

if _HAS_SCOPE:
    class BpmBufferConfig(BasePipelineConfig):
        """Configuration schema for BPM Timecoded Buffer preprocessor."""

        # --- Class attributes (no type annotation = plain class var, not Pydantic field) ---
        pipeline_id = "bpm_sync_timecoded_buffer__vj_tools"
        pipeline_name = "BPM Sync Timecoded Buffer (VJ.Tools)"
        pipeline_description = (
            "Beat-grid timecoding via Ableton Link. Stamps VJSync barcode on "
            "input, preserves through AI via VACE masking (mask=0 at barcode). "
            "Client decodes surviving barcode on output for beat-accurate timing."
        )
        supports_prompts = False
        modified = True
        usage = [UsageType.PREPROCESSOR]
        modes = {
            "video": ModeDefaults(default=True),
            "text": ModeDefaults(default=True),
        }

        # --- Runtime parameters ---

        barcode_height: int = Field(
            default=16,
            ge=4,
            le=128,
            json_schema_extra=ui_field_config(
                order=0,
                label="Barcode Height (px)",
                is_load_param=False,
            ),
        )

        mask_feather: int = Field(
            default=2,
            ge=0,
            le=16,
            json_schema_extra=ui_field_config(
                order=1,
                label="Mask Feather (px)",
                is_load_param=False,
            ),
        )

        test_input: bool = Field(
            default=False,
            json_schema_extra=ui_field_config(
                order=2,
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
            self.barcode_height = kwargs.get("barcode_height", 16)
            self.mask_feather = kwargs.get("mask_feather", 2)
            self.test_input = kwargs.get("test_input", False)


# --- Pipeline ---

class BpmTimecodedBufferPipeline(Pipeline):
    """
    Scope preprocessor that stamps beat-grid timecodes using Ableton Link
    and creates VACE masks to preserve them through AI generation.

    Automatically joins an Ableton Link session on startup. The barcode is
    stamped on input, the VACE mask ensures AI generates everything EXCEPT
    the barcode strip (mask=0 = preserve), and the client decodes the
    surviving barcode on the output for beat-accurate timing.

    VACE mask format (matching Scope's VaceEncodingBlock):
      - vace_input_frames [1,3,F,H,W] [-1,1]: full frames with barcode
        Scope splits internally: inactive = frame*(1-mask), reactive = frame*mask
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

        # Always start Ableton Link — BPM comes from the Link session
        self._clock = LinkClock(120.0)
        self._clock.start(120.0)

        # Test pattern source (lazy-initialized on first use)
        self._test_source: Optional[TestPatternSource] = None

        logger.info(
            f"[BPM Buffer] Pipeline initialized "
            f"(device={self.device}, Link started at 120 BPM)"
        )

    def __del__(self):
        if hasattr(self, "_clock"):
            self._clock.stop()

    def set_bpm(self, bpm: float):
        """Manually set BPM (for RunPod / no-Link scenarios)."""
        bpm = max(20.0, min(999.0, bpm))
        self._clock._tempo = bpm
        if self._test_source:
            self._test_source.set_bpm(bpm)
        logger.info(f"[BPM Buffer] Manual BPM: {bpm:.1f}")

    def prepare(self, **kwargs) -> "Requirements":
        """Tell Scope how many input frames we need per call."""
        return Requirements(input_size=1)

    def __call__(self, **kwargs) -> dict:
        """
        Process input frames (Scope Pipeline interface).

        Scope passes video as kwargs["video"] — a list of (1, H, W, C) uint8 tensors.

        Returns:
            dict with video, vace_input_frames, vace_input_masks
        """
        video = kwargs.get("video", [])
        barcode_h = kwargs.get("barcode_height", getattr(self.config, "barcode_height", 16))
        mask_feather = kwargs.get("mask_feather", getattr(self.config, "mask_feather", 2))
        test_input = kwargs.get("test_input", getattr(self.config, "test_input", False))

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
                self._clock, num_frames=len(video), barcode_height=barcode_h
            )

        # Stack frames
        frames = torch.cat(video, dim=0).float()  # (F, H, W, C), [0, 255]
        F, H, W, C = frames.shape
        barcode_h = min(barcode_h, H // 4)

        # --- 1. Stamp barcode on each frame using Link beat clock ---
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

        # --- 2. Generate VACE mask ---
        # mask=1 -> inpaint (AI generates everything above barcode)
        # mask=0 -> preserve (barcode strip survives AI untouched)
        mask = torch.ones(F, H, W, dtype=torch.float32)
        mask[:, -barcode_h:, :] = 0.0  # Preserve barcode strip

        # Feathering at boundary for smooth transition
        if mask_feather > 0:
            boundary_y = H - barcode_h
            for dy in range(mask_feather):
                if boundary_y - dy - 1 >= 0:
                    alpha = (dy + 1) / (mask_feather + 1)
                    mask[:, boundary_y - dy - 1, :] = alpha

        # --- 3. Build VACE mask ---
        # Only forward the mask, NOT vace_input_frames.
        # Scope's pipeline_processor checks "vace_input_frames" not in call_params
        # before routing video from the queue. If we include vace_input_frames,
        # it blocks the video input for non-VACE pipelines (like passthrough),
        # causing "Input cannot be None" errors.
        #
        # By only providing the mask, Scope will:
        # - Route our stamped video through the queue to the main pipeline
        # - If VACE is enabled, auto-route the video as vace_input_frames
        # - Apply our custom mask (barcode=0=preserve, content=1=generate)
        vace_mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, F, H, W)
        vace_mask = vace_mask.to(device=self.device, dtype=self.dtype)

        # --- 4. Video output ---
        # Return clean stamped frames in [0,1] for the pipeline chain.
        # Pipeline_processor will normalize to uint8 and queue for next pipeline.
        frames_01 = frames_stamped / 255.0  # (F, H, W, C), [0, 1]

        result = {"video": frames_01}

        # Forward VACE mask — the barcode MUST be preserved through AI.
        # The mask ensures AI generates everything except the barcode strip.
        # NOTE: We intentionally do NOT include vace_input_frames here.
        # Scope auto-routes the video queue to vace_input_frames when VACE is enabled.
        result["vace_input_masks"] = vace_mask

        # Link state for diagnostics
        result["_bpm_buffer_meta"] = {
            "beat": self._clock.beat,
            "bpm": self._clock.tempo,
            "link_enabled": self._clock.enabled,
            "frame_seq": self._frame_seq,
        }

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

        # Pydantic fields

        buffer_mode: BufferMode = Field(
            default=BufferMode.LATENCY,
            json_schema_extra=ui_field_config(
                order=0,
                label="Buffer Mode",
                is_load_param=False,
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

        link_sync: bool = Field(
            default=False,
            json_schema_extra=ui_field_config(
                order=5,
                label="Ableton Link Sync",
            ),
        )

        link_bpm: float = Field(
            default=120.0,
            ge=20.0,
            le=999.0,
            json_schema_extra=ui_field_config(
                order=6,
                label="Link BPM",
            ),
        )

        barcode_height: int = Field(
            default=16,
            ge=4,
            le=128,
            json_schema_extra=ui_field_config(
                order=7,
                label="Barcode Height (px)",
                is_load_param=True,
            ),
        )

        buffer_fill_pct: float = Field(
            default=0.0,
            ge=0.0,
            le=100.0,
            json_schema_extra=ui_field_config(
                order=8,
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
            self.link_sync = kwargs.get("link_sync", False)
            self.link_bpm = kwargs.get("link_bpm", 120.0)
            self.barcode_height = kwargs.get("barcode_height", 16)
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

    Buffer modes:
      - no_buffer: Just strip the barcode (pass-through, simplest)
      - latency:   Adjustable latency (default) — FIFO + binary search with
                   configurable delay in ms (0-60000). MIDI fader assignable.
                   Sliding shorter → catch up, sliding longer → buffer fills.
      - beat:      Beat-delayed — delay = beat_buffer_depth × ms_per_beat.
                   Plays back every frame smoothly, delayed by N beats.
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

        # Ableton Link clock for local sync
        self._clock: Optional[LinkClock] = None
        self._link_active = False

        # Auto-start Link if config says so
        link_sync = getattr(config, "link_sync", False)
        if link_sync:
            link_bpm = getattr(config, "link_bpm", 120.0)
            self._start_link(link_bpm)

        # --- Wall-clock FIFO buffers ---
        # Beat mode and latency mode share the same architecture:
        # append all frames with timestamp, binary search for target time
        self._beat_fifo: list[_BufferedFrame] = []
        self._latency_fifo: list[_BufferedFrame] = []

        # Current output frame (held between calls when FIFO is empty)
        self._current_output: Optional[np.ndarray] = None

        # Hold state — freezes target_time at the moment hold was engaged
        self._hold_active: bool = False
        self._hold_target_time: float = 0.0  # frozen wall-clock target

        # Extra delay accumulated during hold (added to beat delay on release)
        self._playback_extra_delay: float = 0.0

        # Stats
        self._decode_success = 0
        self._decode_fail = 0

        logger.info("[BPM Buffer Output] Postprocessor initialized (wall-clock FIFO)")

    def _start_link(self, bpm: float = 120.0):
        """Start Ableton Link clock for local beat sync."""
        if self._clock is not None:
            return
        self._clock = LinkClock(bpm)
        self._clock.start(bpm)
        self._link_active = True
        logger.info(f"[BPM Buffer Output] Ableton Link started at {bpm} BPM")

    def _stop_link(self):
        """Stop Ableton Link clock."""
        if self._clock is not None:
            self._clock.stop()
            self._clock = None
        self._link_active = False
        logger.info("[BPM Buffer Output] Ableton Link stopped")

    def __del__(self):
        if hasattr(self, "_clock") and self._clock is not None:
            self._clock.stop()

    def _get_effective_bpm(self) -> float:
        """Get BPM from Link clock → latest frame barcode → synthetic fallback."""
        if self._link_active and self._clock is not None:
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

        # lo is now the first frame >= target_time
        # Check if the previous frame is actually closer
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
        Done BEFORE binary search to prevent returning stale frames.
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
        barcode_h = kwargs.get("barcode_height", getattr(self.config, "barcode_height", 16))
        mode = str(kwargs.get("buffer_mode", getattr(self.config, "buffer_mode", "latency")))
        latency_ms = kwargs.get("latency_delay_ms", getattr(self.config, "latency_delay_ms", 100))
        beat_depth = kwargs.get("beat_buffer_depth", getattr(self.config, "beat_buffer_depth", 4))
        reset_buffer = kwargs.get("reset_buffer", getattr(self.config, "reset_buffer", False))
        link_sync = kwargs.get("link_sync", getattr(self.config, "link_sync", False))
        hold = kwargs.get("hold", getattr(self.config, "hold", False))

        # --- Ableton Link toggle ---
        if link_sync and not self._link_active:
            link_bpm = kwargs.get("link_bpm", getattr(self.config, "link_bpm", 120.0))
            self._start_link(link_bpm)
        elif not link_sync and self._link_active:
            self._stop_link()

        # --- Reset buffer trigger (MIDI button assignable) ---
        if reset_buffer:
            self._beat_fifo.clear()
            self._latency_fifo.clear()
            self._current_output = None
            self._playback_extra_delay = 0.0
            self._hold_active = False
            logger.info("[BPM Buffer Output] Buffer reset")

        # --- Hold toggle ---
        if hold and not self._hold_active:
            # Engage hold — freeze target_time at current playback position
            self._hold_active = True
            bpm = self._get_effective_bpm()
            ms_per_beat = 60_000.0 / bpm
            delay_s = (beat_depth * ms_per_beat + self._playback_extra_delay) / 1000.0
            self._hold_target_time = time.monotonic() - delay_s
            logger.info(f"[BPM Buffer Output] HOLD engaged at target_time={self._hold_target_time:.3f}")
        elif not hold and self._hold_active:
            # Release hold — the time that passed during hold becomes extra delay
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
        barcode_h = min(barcode_h, H // 4)
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
                # Use Link beat if available, otherwise synthetic
                if self._link_active and self._clock is not None:
                    beat = self._clock.beat
                    bpm = self._clock.tempo
                else:
                    beat = 0.0
                    bpm = self.SYNTHETIC_BPM

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
            beat_fill = len(self._beat_fifo)
            lat_fill = len(self._latency_fifo)
            logger.info(
                f"[BPM Buffer Output] mode={mode}, decode={self._decode_success}/{total}, "
                f"link={self._link_active}, hold={self._hold_active}, "
                f"beat_fifo={beat_fill}, lat_fifo={lat_fill}, "
                f"incoming={len(incoming)}"
            )

        # Convert single output frame to tensor
        out_tensor = torch.from_numpy(output_frame).float().unsqueeze(0) / 255.0  # (1, H, W, C) [0,1]

        result = {"video": out_tensor}

        # --- Buffer fill percentage (output to UI slider) ---
        # Calculate how full the active buffer is relative to MAX_FIFO_FRAMES
        if mode == "latency":
            active_fifo_size = len(self._latency_fifo)
        elif mode == "beat":
            active_fifo_size = len(self._beat_fifo)
        else:
            active_fifo_size = 0
        buffer_fill_pct = min(100.0, (active_fifo_size / self.MAX_FIFO_FRAMES) * 100.0)

        # Write back to config so Scope can display it on the read-only slider
        if hasattr(self.config, "buffer_fill_pct"):
            self.config.buffer_fill_pct = buffer_fill_pct

        # Diagnostics metadata
        total = self._decode_success + self._decode_fail
        rate = self._decode_success / total if total > 0 else 0.0
        bpm = self._get_effective_bpm()
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
            "link_active": self._link_active,
            "link_beat": self._clock.beat if self._clock else None,
            "link_bpm": self._clock.tempo if self._clock else None,
            "link_peers": self._clock.num_peers if self._clock else 0,
        }

        return result

    def _process_beat(
        self,
        incoming: list[_BufferedFrame],
        beat_depth: int,
    ) -> np.ndarray:
        """
        Beat-delayed mode — wall-clock FIFO + binary search.

        All frames are pushed into beat_fifo with their wall-clock timestamp.
        On each call, we compute:
            delay_ms = beat_depth × (60000 / bpm) + playback_extra_delay
            target_time = now - delay_s
        Then binary search for the closest frame to target_time.

        This gives smooth full-framerate playback delayed by exactly N beats.
        No stepped/stutter behavior — every frame plays in order.
        """
        # Ingest: push all incoming frames, hard cap only
        self._beat_fifo.extend(incoming)
        while len(self._beat_fifo) > self.MAX_FIFO_FRAMES:
            self._beat_fifo.pop(0)

        if not self._beat_fifo:
            return self._current_output if self._current_output is not None else incoming[-1].frame

        # Compute target time
        bpm = self._get_effective_bpm()
        ms_per_beat = 60_000.0 / bpm
        delay_ms = beat_depth * ms_per_beat + self._playback_extra_delay
        delay_s = delay_ms / 1000.0

        if self._hold_active and self._hold_target_time > 0:
            target_time = self._hold_target_time
        else:
            target_time = time.monotonic() - delay_s

        # Evict old frames BEFORE search (prevents returning stale data)
        self._evict_old_frames(self._beat_fifo, target_time, keep_margin=2.0)

        if not self._beat_fifo:
            return self._current_output if self._current_output is not None else incoming[-1].frame

        # Binary search for closest frame
        frame = self._binary_search_closest(self._beat_fifo, target_time)
        if frame is not None:
            self._current_output = frame

        return self._current_output if self._current_output is not None else incoming[-1].frame

    def _process_latency(
        self, incoming: list[_BufferedFrame], delay_ms: int
    ) -> np.ndarray:
        """
        Latency compensation mode — wall-clock FIFO + binary search.

        Same architecture as beat mode but with a direct ms delay parameter.
        Sliding the delay shorter → playback speeds up to catch up.
        Sliding the delay longer → playback slows until buffer fills.
        This emergent behavior from continuous slider + binary search is
        smooth and expressive — perfect for MIDI fader control.

        delay_ms range: 0-60000 (0 to 60 seconds)
        """
        # Ingest: push all incoming frames, hard cap only
        self._latency_fifo.extend(incoming)
        while len(self._latency_fifo) > self.MAX_FIFO_FRAMES:
            self._latency_fifo.pop(0)

        if not self._latency_fifo:
            return self._current_output if self._current_output is not None else incoming[-1].frame

        # Compute target time
        delay_s = delay_ms / 1000.0
        target_time = time.monotonic() - delay_s

        # Evict old frames BEFORE search
        self._evict_old_frames(self._latency_fifo, target_time, keep_margin=2.0)

        if not self._latency_fifo:
            return self._current_output if self._current_output is not None else incoming[-1].frame

        # Binary search for closest frame
        frame = self._binary_search_closest(self._latency_fifo, target_time)
        if frame is not None:
            self._current_output = frame

        return self._current_output if self._current_output is not None else incoming[-1].frame
