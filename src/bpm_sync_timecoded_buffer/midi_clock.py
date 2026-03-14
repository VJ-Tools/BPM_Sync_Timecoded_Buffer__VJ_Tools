"""
MIDI Clock — Derives BPM from MIDI clock messages (0xF8).

MIDI clock sends 24 pulses per quarter note (PPQN).
By measuring the interval between 0xF8 messages, we derive BPM:
    BPM = 60 / (avg_interval × 24)

Uses `mido` with `python-rtmidi` backend for cross-platform MIDI input.
Falls back gracefully if mido is not installed.

MIDI message types:
  0xF8 = Timing Clock (24 per quarter note)
  0xFA = Start
  0xFB = Continue
  0xFC = Stop
"""

import collections
import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Number of clock intervals to average for BPM calculation
_CLOCK_WINDOW = 48  # 2 beats worth of clocks for smooth averaging
_PPQN = 24  # Pulses per quarter note (MIDI standard)


class MidiClock:
    """
    Listens for MIDI clock messages on a named port and derives BPM + beat position.

    Usage:
        clock = MidiClock()
        devices = MidiClock.list_devices()
        clock.start(device_name=devices[0])
        print(clock.tempo, clock.beat)
        clock.stop()
    """

    def __init__(self):
        self._tempo: float = 0.0  # Derived BPM (0 = no clock received yet)
        self._beat: float = 0.0
        self._phase: float = 0.0
        self._enabled: bool = False
        self._running: bool = False  # True after MIDI Start, False after MIDI Stop
        self._device_name: str = ""

        # Thread management
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Clock interval tracking
        self._last_clock_time: float = 0.0
        self._clock_intervals: collections.deque = collections.deque(maxlen=_CLOCK_WINDOW)
        self._pulse_count: int = 0  # Total pulses since Start

    @staticmethod
    def list_devices() -> list[str]:
        """List available MIDI input devices."""
        try:
            import mido
            return mido.get_input_names()
        except ImportError:
            logger.warning("[MIDI Clock] mido not installed. Install with: pip install mido python-rtmidi")
            return []
        except Exception as e:
            logger.error(f"[MIDI Clock] Error listing devices: {e}")
            return []

    def start(self, device_name: str = ""):
        """Start listening for MIDI clock on the given device."""
        if self._thread is not None:
            self.stop()

        self._device_name = device_name
        self._stop_event.clear()
        self._clock_intervals.clear()
        self._last_clock_time = 0.0
        self._pulse_count = 0
        self._beat = 0.0
        self._phase = 0.0
        self._tempo = 0.0

        self._thread = threading.Thread(
            target=self._listen_loop, daemon=True, name="midi-clock"
        )
        self._thread.start()
        logger.info(f"[MIDI Clock] Started listening on '{device_name or 'default'}'")

    def stop(self):
        """Stop listening for MIDI clock."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._enabled = False
        self._running = False
        logger.info("[MIDI Clock] Stopped")

    def _listen_loop(self):
        """Background thread: listen for MIDI messages."""
        try:
            import mido
        except ImportError:
            logger.error(
                "[MIDI Clock] mido not installed — cannot receive MIDI clock. "
                "Install with: pip install mido python-rtmidi"
            )
            return

        port = None
        try:
            if self._device_name:
                port = mido.open_input(self._device_name)
            else:
                # Open default/first available
                names = mido.get_input_names()
                if not names:
                    logger.error("[MIDI Clock] No MIDI input devices found")
                    return
                port = mido.open_input(names[0])
                self._device_name = names[0]

            self._enabled = True
            logger.info(f"[MIDI Clock] Connected to '{self._device_name}'")

            while not self._stop_event.is_set():
                # Non-blocking poll with timeout
                for msg in port.iter_pending():
                    self._handle_message(msg)

                # Small sleep to avoid busy-wait while staying responsive
                time.sleep(0.001)  # 1ms — well under MIDI clock resolution

        except Exception as e:
            logger.error(f"[MIDI Clock] Error: {e}")
        finally:
            if port is not None:
                try:
                    port.close()
                except Exception:
                    pass
            self._enabled = False

    def _handle_message(self, msg):
        """Handle a single MIDI message."""
        if msg.type == "clock":
            # 0xF8 — Timing Clock
            now = time.monotonic()

            if self._last_clock_time > 0:
                interval = now - self._last_clock_time
                # Sanity check: reject intervals that imply < 20 BPM or > 999 BPM
                if 0.0025 < interval < 0.125:  # ~20-999 BPM range
                    self._clock_intervals.append(interval)

            self._last_clock_time = now
            self._pulse_count += 1

            # Update beat position (24 pulses = 1 beat)
            if self._running:
                self._beat = self._pulse_count / _PPQN
                self._phase = self._beat % 4.0

            # Recalculate BPM from averaged intervals
            if len(self._clock_intervals) >= 6:  # Need at least 6 for stable reading
                avg_interval = sum(self._clock_intervals) / len(self._clock_intervals)
                self._tempo = 60.0 / (avg_interval * _PPQN)

        elif msg.type == "start":
            # 0xFA — Start
            self._running = True
            self._pulse_count = 0
            self._beat = 0.0
            self._phase = 0.0
            logger.info("[MIDI Clock] MIDI Start received")

        elif msg.type == "continue":
            # 0xFB — Continue
            self._running = True
            logger.info("[MIDI Clock] MIDI Continue received")

        elif msg.type == "stop":
            # 0xFC — Stop
            self._running = False
            logger.info("[MIDI Clock] MIDI Stop received")

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
    def running(self) -> bool:
        """True if MIDI transport is playing (after Start/Continue, before Stop)."""
        return self._running

    @property
    def device_name(self) -> str:
        return self._device_name
