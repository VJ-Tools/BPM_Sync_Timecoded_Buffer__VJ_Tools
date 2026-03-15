"""
Audio-Reactive Scope Controller
================================
Captures microphone audio, analyzes frequency bands and beats in real-time,
and sends OSC messages to Daydream Scope to drive AI video generation.

Usage:
    python audio_reactive_scope.py                    # Use default mic
    python audio_reactive_scope.py --device 7         # Use LoopBeAudio (system audio)
    python audio_reactive_scope.py --list-devices     # List audio devices
    python audio_reactive_scope.py --port 52178       # Custom Scope port

Controls (keyboard while running):
    1-5     Select prompt preset
    +/-     Adjust sensitivity
    b       Force beat / cache reset
    h       Toggle hold
    q       Quit
"""

import argparse
import math
import sys
import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd
from pythonosc.udp_client import SimpleUDPClient


# ─── Audio Analysis ──────────────────────────────────────────────────────────

class AudioAnalyzer:
    """Real-time audio frequency band analysis and onset detection."""

    def __init__(self, samplerate=44100, block_size=1024):
        self.samplerate = samplerate
        self.block_size = block_size

        # Frequency band edges (Hz)
        self.band_edges = [20, 150, 400, 2500, 8000, 20000]
        self.band_names = ["sub", "bass", "mid", "high_mid", "high"]

        # Smoothed energy per band
        self.band_energy = np.zeros(len(self.band_names), dtype=np.float64)
        self.band_peak = np.zeros(len(self.band_names), dtype=np.float64)

        # Overall RMS
        self.rms = 0.0
        self.peak_rms = 0.001

        # Beat detection
        self.onset_history = deque(maxlen=100)
        self.energy_history = deque(maxlen=50)
        self.beat_detected = False
        self.last_beat_time = 0.0
        self.beat_cooldown = 0.15  # seconds between beats
        self.bpm_estimate = 120.0

        # Sensitivity
        self.sensitivity = 1.0

        # Pre-compute FFT bin → band mapping
        freqs = np.fft.rfftfreq(block_size, 1.0 / samplerate)
        self.band_masks = []
        for i in range(len(self.band_names)):
            lo, hi = self.band_edges[i], self.band_edges[i + 1]
            mask = (freqs >= lo) & (freqs < hi)
            self.band_masks.append(mask)

    def process(self, audio_block: np.ndarray):
        """Process a block of audio samples. audio_block shape: (block_size,) or (block_size, channels)."""
        if audio_block.ndim > 1:
            audio_block = audio_block.mean(axis=1)

        # RMS
        self.rms = float(np.sqrt(np.mean(audio_block ** 2))) * self.sensitivity
        self.peak_rms = max(self.peak_rms * 0.9995, self.rms, 0.001)

        # FFT
        windowed = audio_block * np.hanning(len(audio_block))
        spectrum = np.abs(np.fft.rfft(windowed))

        # Band energies (normalized)
        for i, mask in enumerate(self.band_masks):
            if mask.any():
                raw = float(np.mean(spectrum[mask] ** 2))
                smoothed = self.band_energy[i] * 0.7 + raw * 0.3
                self.band_energy[i] = smoothed
                self.band_peak[i] = max(self.band_peak[i] * 0.999, smoothed, 0.0001)

        # Onset detection (spectral flux in bass/sub region)
        bass_energy = float(self.band_energy[0] + self.band_energy[1])
        self.energy_history.append(bass_energy)

        self.beat_detected = False
        if len(self.energy_history) > 8:
            avg = np.mean(list(self.energy_history)[:-1])
            now = time.monotonic()
            threshold = avg * (1.3 / self.sensitivity)
            if bass_energy > threshold and (now - self.last_beat_time) > self.beat_cooldown:
                self.beat_detected = True
                dt = now - self.last_beat_time
                self.last_beat_time = now
                if 0.2 < dt < 2.0:
                    instant_bpm = 60.0 / dt
                    self.bpm_estimate = self.bpm_estimate * 0.8 + instant_bpm * 0.2

    def get_normalized_bands(self) -> dict:
        """Get band energies normalized to 0.0-1.0."""
        result = {}
        for i, name in enumerate(self.band_names):
            result[name] = float(
                min(1.0, self.band_energy[i] / self.band_peak[i])
                if self.band_peak[i] > 0 else 0.0
            )
        return result

    @property
    def normalized_rms(self) -> float:
        return min(1.0, self.rms / self.peak_rms) if self.peak_rms > 0 else 0.0


# ─── Scope OSC Controller ────────────────────────────────────────────────────

class ScopeController:
    """Maps audio analysis to Scope OSC parameters."""

    PROMPTS = [
        "abstract flowing energy, neon particles, cosmic nebula, vibrant colors",
        "underwater bioluminescent creatures, deep ocean, jellyfish, coral reef",
        "cyberpunk cityscape, rain, neon signs, reflections, blade runner",
        "psychedelic fractal patterns, kaleidoscope, sacred geometry, vivid",
        "fire and smoke, volcanic eruption, magma flow, ember particles, dramatic",
        "crystal cave, prismatic light, refraction, gemstones, ethereal glow",
        "aurora borealis, northern lights, starfield, cosmic dance, atmospheric",
        "glitch art, digital corruption, vaporwave, retro CRT, data moshing",
    ]

    def __init__(self, host="127.0.0.1", port=52178):
        self.client = SimpleUDPClient(host, port)
        self.current_prompt_idx = 0
        self.hold = False

        # Mapping parameters (tweak these for different feels)
        self.noise_floor = 0.15       # minimum noise_scale
        self.noise_ceiling = 0.75     # maximum noise_scale
        self.strength_floor = 0.3     # minimum strength
        self.strength_ceiling = 0.85  # maximum strength
        self.cache_bias_floor = 0.15  # low coherence (more change)
        self.cache_bias_ceiling = 0.8 # high coherence (stable)

        # Smoothing
        self._smooth_noise = 0.3
        self._smooth_strength = 0.5
        self._smooth_cache_bias = 0.5
        self._smooth_vace = 1.0

        # Beat tracking
        self._beat_burst = 0.0
        self._cache_reset_cooldown = 0.0

    def update(self, analyzer: AudioAnalyzer):
        """Send OSC updates based on current audio analysis."""
        bands = analyzer.get_normalized_bands()
        rms = analyzer.normalized_rms

        # ── noise_scale: driven by bass + sub (more bass = more transformation)
        bass_drive = (bands["sub"] * 0.6 + bands["bass"] * 0.4)
        target_noise = self.noise_floor + bass_drive * (self.noise_ceiling - self.noise_floor)

        # Beat burst: spike noise on beat
        if analyzer.beat_detected:
            self._beat_burst = 0.3
        self._beat_burst *= 0.85
        target_noise = min(1.0, target_noise + self._beat_burst)

        self._smooth_noise += (target_noise - self._smooth_noise) * 0.3
        self.client.send_message("/scope/noise_scale", float(self._smooth_noise))

        # ── strength: driven by overall energy
        target_strength = self.strength_floor + rms * (self.strength_ceiling - self.strength_floor)
        self._smooth_strength += (target_strength - self._smooth_strength) * 0.2
        self.client.send_message("/scope/strength", float(self._smooth_strength))

        # ── kv_cache_attention_bias: inverse of high frequencies
        # More highs = less temporal coherence (more change)
        high_drive = bands["high"] * 0.5 + bands["high_mid"] * 0.5
        target_cache = self.cache_bias_ceiling - high_drive * (self.cache_bias_ceiling - self.cache_bias_floor)
        self._smooth_cache_bias += (target_cache - self._smooth_cache_bias) * 0.15
        self.client.send_message("/scope/kv_cache_attention_bias", float(max(0.01, self._smooth_cache_bias)))

        # ── vace_context_scale: driven by mid frequencies
        target_vace = 0.5 + bands["mid"] * 1.0
        self._smooth_vace += (target_vace - self._smooth_vace) * 0.2
        self.client.send_message("/scope/vace_context_scale", float(self._smooth_vace))

        # ── cache reset on very strong beats (drops / impacts)
        self._cache_reset_cooldown = max(0, self._cache_reset_cooldown - 1)
        if analyzer.beat_detected and bass_drive > 0.85 and self._cache_reset_cooldown <= 0:
            self.client.send_message("/scope/reset_cache", True)
            self._cache_reset_cooldown = 30  # ~1 second cooldown at 30fps update rate

    def set_prompt(self, idx: int):
        """Switch to a numbered prompt preset."""
        self.current_prompt_idx = idx % len(self.PROMPTS)
        prompt = self.PROMPTS[self.current_prompt_idx]
        self.client.send_message("/scope/prompt", prompt)
        return prompt

    def toggle_hold(self):
        self.hold = not self.hold
        self.client.send_message("/scope/hold", self.hold)
        return self.hold

    def force_reset(self):
        self.client.send_message("/scope/reset_cache", True)


# ─── Console Visualizer ──────────────────────────────────────────────────────

def render_meter(label: str, value: float, width: int = 30) -> str:
    filled = int(value * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"  {label:>8s} [{bar}] {value:.2f}"


def render_status(analyzer: AudioAnalyzer, controller: ScopeController) -> str:
    bands = analyzer.get_normalized_bands()
    lines = [
        "",
        "  ┌─────────────────────────────────────────────────┐",
        "  │  AUDIO-REACTIVE SCOPE CONTROLLER                │",
        "  └─────────────────────────────────────────────────┘",
        "",
        "  Audio Input:",
        render_meter("SUB", bands.get("sub", 0)),
        render_meter("BASS", bands.get("bass", 0)),
        render_meter("MID", bands.get("mid", 0)),
        render_meter("HI-MID", bands.get("high_mid", 0)),
        render_meter("HIGH", bands.get("high", 0)),
        render_meter("RMS", analyzer.normalized_rms),
        "",
        f"  Beat: {'>>> BEAT <<<' if analyzer.beat_detected else '            '}"
        f"   BPM: {analyzer.bpm_estimate:.0f}"
        f"   Sensitivity: {analyzer.sensitivity:.1f}x",
        "",
        "  Scope Parameters:",
        render_meter("noise", controller._smooth_noise),
        render_meter("strength", controller._smooth_strength),
        render_meter("cache", controller._smooth_cache_bias),
        render_meter("vace", controller._smooth_vace / 2.0),
        "",
        f"  Prompt [{controller.current_prompt_idx + 1}/{len(controller.PROMPTS)}]: "
        f"{controller.PROMPTS[controller.current_prompt_idx][:50]}...",
        f"  Hold: {'ON' if controller.hold else 'OFF'}",
        "",
        "  Keys: [1-8] prompts  [+/-] sensitivity  [b] beat  [h] hold  [q] quit",
    ]
    return "\n".join(lines)


# ─── Main Loop ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Audio-reactive Scope controller")
    parser.add_argument("--device", type=int, default=None, help="Audio input device index")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("--host", default="127.0.0.1", help="Scope host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=52178, help="Scope OSC port (default: 52178)")
    parser.add_argument("--samplerate", type=int, default=44100, help="Audio sample rate")
    parser.add_argument("--block-size", type=int, default=1024, help="Audio block size")
    parser.add_argument("--sensitivity", type=float, default=1.0, help="Initial sensitivity")
    parser.add_argument("--fps", type=int, default=30, help="OSC update rate")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    # Init
    analyzer = AudioAnalyzer(samplerate=args.samplerate, block_size=args.block_size)
    analyzer.sensitivity = args.sensitivity
    controller = ScopeController(host=args.host, port=args.port)

    # Audio callback
    def audio_callback(indata, frames, time_info, status):
        if status:
            pass  # ignore overflows silently
        analyzer.process(indata[:, 0] if indata.ndim > 1 else indata)

    # Start audio stream
    try:
        stream = sd.InputStream(
            device=args.device,
            channels=1,
            samplerate=args.samplerate,
            blocksize=args.block_size,
            callback=audio_callback,
        )
        stream.start()
    except Exception as e:
        print(f"Failed to open audio device: {e}")
        print("\nAvailable devices:")
        print(sd.query_devices())
        return

    device_info = sd.query_devices(args.device or sd.default.device[0])
    print(f"Audio device: {device_info['name']}")
    print(f"Scope OSC: {args.host}:{args.port}")
    print(f"Update rate: {args.fps} Hz")

    # Set initial prompt
    prompt = controller.set_prompt(0)
    print(f"Prompt: {prompt}")

    # Keyboard input thread
    key_pressed = {"key": None}
    running = True

    def key_listener():
        """Non-blocking key listener for Windows."""
        try:
            import msvcrt
            while running:
                if msvcrt.kbhit():
                    ch = msvcrt.getch().decode("utf-8", errors="ignore")
                    key_pressed["key"] = ch
                time.sleep(0.05)
        except ImportError:
            # Unix fallback
            import tty
            import termios
            import select
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            try:
                tty.setcbreak(fd)
                while running:
                    if select.select([sys.stdin], [], [], 0.05)[0]:
                        key_pressed["key"] = sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)

    key_thread = threading.Thread(target=key_listener, daemon=True)
    key_thread.start()

    # Main control loop
    update_interval = 1.0 / args.fps
    try:
        while True:
            t0 = time.monotonic()

            # Handle keyboard
            key = key_pressed.get("key")
            if key:
                key_pressed["key"] = None
                if key == "q":
                    break
                elif key in "12345678":
                    prompt = controller.set_prompt(int(key) - 1)
                    print(f"\nPrompt {key}: {prompt}")
                elif key == "+":
                    analyzer.sensitivity = min(5.0, analyzer.sensitivity + 0.1)
                elif key == "-":
                    analyzer.sensitivity = max(0.1, analyzer.sensitivity - 0.1)
                elif key == "b":
                    controller.force_reset()
                    print("\n>>> MANUAL CACHE RESET <<<")
                elif key == "h":
                    state = controller.toggle_hold()
                    print(f"\nHold: {'ON' if state else 'OFF'}")

            # Update Scope
            controller.update(analyzer)

            # Render status
            status = render_status(analyzer, controller)
            # Clear screen and redraw
            sys.stdout.write("\033[H\033[J" + status)
            sys.stdout.flush()

            # Maintain update rate
            elapsed = time.monotonic() - t0
            if elapsed < update_interval:
                time.sleep(update_interval - elapsed)

    except KeyboardInterrupt:
        pass
    finally:
        running = False
        stream.stop()
        stream.close()
        print("\nStopped.")


if __name__ == "__main__":
    main()
