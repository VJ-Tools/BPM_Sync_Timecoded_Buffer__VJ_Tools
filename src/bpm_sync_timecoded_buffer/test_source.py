"""
Test Pattern Generator — Python port of VJToolsRealtime's testPatternGenerator.ts

Synthetic beat-mapped video source for validating the VJSync signal chain
without external hardware. Generates a bouncing sphere, beat flash,
color cycling, and burned-in timing text.

Produces numpy frames (H, W, 3) uint8 that feed into the pipeline.

Beat source: LinkClock (Ableton Link or free-running fallback)
"""

import math
import time
import numpy as np
import cv2
from typing import Optional

import torch


def generate_test_frame(
    width: int,
    height: int,
    beat_whole: int,
    beat_frac: float,
    bpm: float,
    bar: int,
    beat_in_bar: int,
    phrase_bar: int,
    phrase_length: int,
    frame_seq: int,
    using_external_clock: bool = False,
    barcode_height: int = 16,
) -> np.ndarray:
    """
    Generate a single test pattern frame matching the TypeScript version.

    Returns: numpy array (H, W, 3), dtype uint8
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # --- Background: hue cycles over 16 bars ---
    hue = ((bar % 16) / 16.0) * 180.0  # OpenCV hue is 0-179
    saturation = int(25 * 2.55)  # ~64
    lightness = int(12 * 2.55)   # ~31

    bg_color = np.array([[[hue, saturation, lightness]]], dtype=np.uint8)
    bg_rgb = cv2.cvtColor(bg_color, cv2.COLOR_HLS2BGR)
    bg_rgb = cv2.cvtColor(bg_rgb, cv2.COLOR_BGR2RGB)
    frame[:, :] = bg_rgb[0, 0]

    content_h = height - barcode_height

    # --- Grid lines ---
    grid_hue = hue
    grid_color = _hsl_to_rgb(grid_hue / 180.0, 0.30, 0.25, alpha=0.3)
    for i in range(1, 4):
        x = int((width / 4) * i)
        if 0 <= x < width:
            _blend_vertical_line(frame, x, 0, content_h, grid_color)

        y = int((height / 4) * i)
        if 0 <= y < content_h:
            _blend_horizontal_line(frame, y, 0, width, grid_color)

    # --- Bouncing sphere ---
    center_x = width // 2
    base_y = int(content_h * 0.45)
    amplitude = int(content_h * 0.2)

    # Bounce: sine wave at beat rate, peaks at beatFrac = 0 (on the beat)
    bounce_y = base_y - int(abs(math.sin(beat_frac * math.pi)) * amplitude)

    # Pulse size on downbeat
    is_downbeat = beat_in_bar == 0 and beat_frac < 0.15
    base_radius = max(15, min(width, content_h) // 15)
    if is_downbeat:
        pulse_radius = base_radius * (1.5 - beat_frac * 3)
    else:
        pulse_radius = base_radius * (1.0 + (1 - beat_frac) * 0.1)
    radius = int(max(base_radius * 0.8, pulse_radius))

    # Sphere hue: complementary to background
    sphere_hue = ((bar % 16) / 16.0 + 0.5) % 1.0

    # Draw sphere with gradient effect
    sphere_bright = _hsl_to_rgb(sphere_hue, 0.80, 0.70)
    sphere_mid = _hsl_to_rgb(sphere_hue, 0.70, 0.50)
    sphere_dark = _hsl_to_rgb(sphere_hue, 0.60, 0.30)

    _draw_gradient_circle(frame, center_x, bounce_y, radius,
                          sphere_bright, sphere_mid, sphere_dark, content_h)

    # Highlight
    hl_x = center_x - radius // 4
    hl_y = bounce_y - radius // 4
    hl_r = max(3, radius // 5)
    _draw_filled_circle(frame, hl_x, hl_y, hl_r,
                        (255, 255, 255), alpha=0.4, max_y=content_h)

    # --- Beat flash bar at bottom of content area ---
    flash_height = 8
    flash_y = content_h - flash_height - 2

    brightness = max(0.0, (1 - beat_frac) * 0.8)
    final_brightness = brightness * 1.0 if beat_in_bar == 0 else brightness * 0.6
    flash_color = (
        int(255 * final_brightness),
        int(255 * final_brightness),
        int(255 * final_brightness),
    )
    if final_brightness > 0.02:
        cv2.rectangle(frame, (10, flash_y), (width - 10, flash_y + flash_height),
                      flash_color, -1)

    # Beat position indicators (4 dots)
    for i in range(4):
        dot_x = int(10 + ((width - 20) / 4) * (i + 0.5))
        dot_y = flash_y + flash_height // 2
        dot_r = 4 if i == beat_in_bar else 2
        dot_alpha = 1.0 if i == beat_in_bar else 0.3
        _draw_filled_circle(frame, dot_x, dot_y, dot_r,
                            (255, 255, 255), alpha=dot_alpha, max_y=content_h)

    # --- Phrase progress bar ---
    bar_y = int(content_h * 0.78)
    bar_height = 6
    bar_x = 12
    bar_w = width - 24

    # Background
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_height),
                  (30, 30, 30), -1)

    # Phrase progress
    phrase_progress = (phrase_bar + (beat_in_bar + beat_frac) / 4.0) / max(1, phrase_length)
    progress_w = int(min(1.0, max(0.0, phrase_progress)) * bar_w)
    progress_hue = ((bar % 16) / 16.0 + 0.5) % 1.0
    progress_color = _hsl_to_rgb(progress_hue, 0.70, 0.55)
    if progress_w > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_w, bar_y + bar_height),
                      progress_color, -1)

    # Bar division markers
    for i in range(1, phrase_length):
        marker_x = bar_x + int((i / phrase_length) * bar_w)
        cv2.line(frame, (marker_x, bar_y), (marker_x, bar_y + bar_height),
                 (80, 80, 80), 1)

    # --- Timing text ---
    text_color = (220, 220, 220)
    muted_color = (130, 130, 130)

    # Top-left: BPM and beat
    cv2.putText(frame, f"BPM: {bpm:.1f}", (12, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Beat: {beat_whole}.{int(beat_frac * 100):02d}", (12, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Bar: {bar}  [{beat_in_bar + 1}/4]", (12, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Phrase: bar {phrase_bar + 1}/{phrase_length}", (12, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, muted_color, 1, cv2.LINE_AA)

    # Top-right: clock source
    clock_label = "EXT CLOCK" if using_external_clock else "INT CLOCK"
    label_size = cv2.getTextSize(clock_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
    cv2.putText(frame, clock_label, (width - label_size[0] - 12, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, muted_color, 1, cv2.LINE_AA)

    # Frame seq
    seq_text = f"SEQ: {frame_seq}"
    seq_size = cv2.getTextSize(seq_text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
    cv2.putText(frame, seq_text, (width - seq_size[0] - 12, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, muted_color, 1, cv2.LINE_AA)

    # Bottom-left label
    cv2.putText(frame, "VJSYNC TEST PATTERN", (12, content_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (90, 90, 90), 1, cv2.LINE_AA)

    return frame


# --- Drawing helpers ---

def _hsl_to_rgb(h: float, s: float, l: float, alpha: float = 1.0):
    """Convert HSL (0-1 range) to RGB tuple (0-255)."""
    if s == 0:
        v = int(l * 255)
        return (v, v, v)

    def hue_to_rgb(p, q, t):
        if t < 0: t += 1
        if t > 1: t -= 1
        if t < 1/6: return p + (q - p) * 6 * t
        if t < 1/2: return q
        if t < 2/3: return p + (q - p) * (2/3 - t) * 6
        return p

    q = l * (1 + s) if l < 0.5 else l + s - l * s
    p = 2 * l - q
    r = hue_to_rgb(p, q, h + 1/3)
    g = hue_to_rgb(p, q, h)
    b = hue_to_rgb(p, q, h - 1/3)

    return (int(r * 255 * alpha), int(g * 255 * alpha), int(b * 255 * alpha))


def _draw_filled_circle(
    frame: np.ndarray, cx: int, cy: int, radius: int,
    color, alpha: float = 1.0, max_y: Optional[int] = None,
):
    """Draw a filled circle with optional alpha blending."""
    if max_y is None:
        max_y = frame.shape[0]
    overlay = frame.copy()
    cv2.circle(overlay, (cx, cy), radius, color, -1, cv2.LINE_AA)
    # Clip to content region
    if alpha < 1.0:
        frame[:max_y] = cv2.addWeighted(
            overlay[:max_y], alpha, frame[:max_y], 1 - alpha, 0
        )
    else:
        frame[:max_y] = overlay[:max_y]


def _draw_gradient_circle(
    frame: np.ndarray, cx: int, cy: int, radius: int,
    bright, mid, dark, max_y: int,
):
    """Draw a sphere with 3-color radial gradient approximation."""
    # Outer ring (dark)
    _draw_filled_circle(frame, cx, cy, radius, dark, max_y=max_y)
    # Mid ring
    _draw_filled_circle(frame, cx, cy, int(radius * 0.7), mid, max_y=max_y)
    # Inner bright
    inner_cx = cx - radius // 5
    inner_cy = cy - radius // 5
    _draw_filled_circle(frame, inner_cx, inner_cy, int(radius * 0.4), bright, max_y=max_y)


def _blend_vertical_line(frame, x, y_start, y_end, color):
    """Draw a semi-transparent vertical line."""
    if 0 <= x < frame.shape[1]:
        r, g, b = color
        alpha = 0.3
        frame[y_start:y_end, x, 0] = np.clip(
            frame[y_start:y_end, x, 0] * (1 - alpha) + r * alpha, 0, 255
        ).astype(np.uint8)
        frame[y_start:y_end, x, 1] = np.clip(
            frame[y_start:y_end, x, 1] * (1 - alpha) + g * alpha, 0, 255
        ).astype(np.uint8)
        frame[y_start:y_end, x, 2] = np.clip(
            frame[y_start:y_end, x, 2] * (1 - alpha) + b * alpha, 0, 255
        ).astype(np.uint8)


def _blend_horizontal_line(frame, y, x_start, x_end, color):
    """Draw a semi-transparent horizontal line."""
    if 0 <= y < frame.shape[0]:
        r, g, b = color
        alpha = 0.3
        frame[y, x_start:x_end, 0] = np.clip(
            frame[y, x_start:x_end, 0] * (1 - alpha) + r * alpha, 0, 255
        ).astype(np.uint8)
        frame[y, x_start:x_end, 1] = np.clip(
            frame[y, x_start:x_end, 1] * (1 - alpha) + g * alpha, 0, 255
        ).astype(np.uint8)
        frame[y, x_start:x_end, 2] = np.clip(
            frame[y, x_start:x_end, 2] * (1 - alpha) + b * alpha, 0, 255
        ).astype(np.uint8)


class TestPatternSource:
    """
    Self-contained test pattern generator — Python port of
    VJToolsRealtime's testPatternGenerator.ts.

    Generates beat-synced bouncing ball frames using the pipeline's
    LinkClock or a manual BPM tap.

    Usage inside the pipeline:
        source = TestPatternSource(width=576, height=336)
        frames = source.generate_batch(clock, num_frames=4)

    Usage standalone with tap tempo:
        source = TestPatternSource(width=576, height=336)
        source.set_bpm(128.0)    # manual BPM
        source.tap()             # or tap in the tempo
        source.tap()
        source.tap()
        frames = source.generate_batch_freerunning(num_frames=4)
    """

    def __init__(self, width: int = 576, height: int = 336):
        self.width = width
        self.height = height
        self._frame_seq = 0

        # Internal free-running clock for when no LinkClock is available
        self._internal_bpm = 120.0
        self._internal_beat = 0.0
        self._last_tick = time.monotonic()

        # Tap tempo state
        self._tap_times: list[float] = []
        self._tap_window = 8  # keep last N taps

    def set_bpm(self, bpm: float):
        """Manually set BPM (for RunPod / no-Link scenarios)."""
        self._internal_bpm = max(20.0, min(999.0, bpm))

    def tap(self) -> Optional[float]:
        """
        Tap tempo. Call this repeatedly to tap in a BPM.
        Returns the detected BPM after 2+ taps, or None if only 1 tap.

        Usage:
            source.tap()   # first tap — starts counting
            source.tap()   # second tap — returns detected BPM
            source.tap()   # third tap — refines BPM average
        """
        now = time.monotonic()
        self._tap_times.append(now)

        # Keep only recent taps (expire taps older than 4 seconds)
        self._tap_times = [
            t for t in self._tap_times
            if now - t < 4.0
        ]

        # Trim to window size
        if len(self._tap_times) > self._tap_window:
            self._tap_times = self._tap_times[-self._tap_window:]

        if len(self._tap_times) < 2:
            return None

        # Average interval between consecutive taps
        intervals = [
            self._tap_times[i] - self._tap_times[i - 1]
            for i in range(1, len(self._tap_times))
        ]
        avg_interval = sum(intervals) / len(intervals)

        if avg_interval > 0:
            detected_bpm = 60.0 / avg_interval
            self._internal_bpm = max(20.0, min(999.0, detected_bpm))
            return self._internal_bpm

        return None

    def _advance_internal_clock(self):
        """Advance the free-running internal clock."""
        now = time.monotonic()
        elapsed = now - self._last_tick
        self._last_tick = now

        # Clamp to prevent jumps
        elapsed = min(elapsed, 0.2)
        beats_elapsed = elapsed * (self._internal_bpm / 60.0)
        self._internal_beat += beats_elapsed

    def generate_frame(
        self,
        beat: float,
        bpm: float,
        barcode_height: int = 16,
    ) -> torch.Tensor:
        """Generate a single test frame as a (1, H, W, 3) uint8 tensor."""
        beat_whole = int(beat) & 0xFFF
        beat_frac = beat - int(beat)
        bar = beat_whole // 4
        beat_in_bar = beat_whole % 4
        phrase_bar = bar % 8
        phrase_length = 8

        frame_np = generate_test_frame(
            self.width, self.height,
            beat_whole=beat_whole,
            beat_frac=beat_frac,
            bpm=bpm,
            bar=bar,
            beat_in_bar=beat_in_bar,
            phrase_bar=phrase_bar,
            phrase_length=phrase_length,
            frame_seq=self._frame_seq,
            using_external_clock=False,
            barcode_height=barcode_height,
        )
        self._frame_seq += 1
        return torch.from_numpy(frame_np).unsqueeze(0)

    def generate_batch(
        self,
        clock,
        num_frames: int = 4,
        barcode_height: int = 16,
    ) -> list:
        """
        Generate a batch of test frames from a LinkClock.

        Args:
            clock: LinkClock instance (has .beat and .tempo properties)
            num_frames: Number of frames to generate
            barcode_height: Space reserved for barcode strip

        Returns:
            List of (1, H, W, 3) uint8 tensors
        """
        frames = []
        beat = clock.beat
        bpm = clock.tempo
        beat_per_frame = (bpm / 60.0) / 30.0  # ~30fps
        for i in range(num_frames):
            frame_beat = beat + i * beat_per_frame
            frames.append(self.generate_frame(frame_beat, bpm, barcode_height))
        return frames

    def generate_batch_freerunning(
        self,
        num_frames: int = 4,
        barcode_height: int = 16,
    ) -> list:
        """
        Generate a batch using the internal free-running clock.
        Use this on RunPod or anywhere without Ableton Link.

        Set BPM with set_bpm() or tap() before calling.

        Returns:
            List of (1, H, W, 3) uint8 tensors
        """
        self._advance_internal_clock()
        frames = []
        beat = self._internal_beat
        bpm = self._internal_bpm
        beat_per_frame = (bpm / 60.0) / 30.0
        for i in range(num_frames):
            frame_beat = beat + i * beat_per_frame
            frames.append(self.generate_frame(frame_beat, bpm, barcode_height))
        return frames
