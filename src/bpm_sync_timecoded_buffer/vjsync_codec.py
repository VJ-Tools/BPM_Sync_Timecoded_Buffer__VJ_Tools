"""
VJSync Barcode Encoder/Decoder

Encodes beat-grid timecode into a 16px barcode strip using BCH(71,50,3)
error correction. Compatible with the TypeScript vjsync package.

Barcode spec:
  - 85 bars total: 14 sync + 71 data/ECC
  - 6px per bar, 16px tall, ITU-R limited range (Y=16 black, Y=235 white)
  - Payload: beatWhole(12b) + beatFrac(8b) + frameSeq(14b) + bpm(9b) + flags(7b) = 50 bits
  - BCH(71,50,3): corrects up to 3 bit errors per frame
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

# --- Constants ---

BAR_WIDTH = 6
STRIP_HEIGHT = 16
BLACK_LEVEL = 16
WHITE_LEVEL = 235

SYNC_BARS = 14
DATA_BARS = 71
TOTAL_BARS = SYNC_BARS + DATA_BARS  # 85

PAYLOAD_BITS = 50
ECC_BITS = 21

# Sync patterns (bar values: 0=black, 1=white)
SYNC_START = [0, 1, 0, 1]
SYNC_CLOCK = [1, 0, 1, 0, 1, 0]
SYNC_GUARD = [0, 1, 0, 1]
SYNC_PATTERN = SYNC_START + SYNC_CLOCK + SYNC_GUARD  # 14 bars

# Field layout
FIELD_BEAT_WHOLE = (0, 12)   # offset, bits
FIELD_BEAT_FRAC = (12, 8)
FIELD_FRAME_SEQ = (20, 14)
FIELD_BPM = (34, 9)
FIELD_FLAGS = (43, 7)


@dataclass
class VJSyncPayload:
    """Payload encoded into the barcode strip (50 bits)."""
    beat_whole: int = 0    # 0-4095 (12 bits)
    beat_frac: int = 0     # 0-255 (8 bits)
    frame_seq: int = 0     # 0-16383 (14 bits)
    bpm_encoded: int = 0   # 0-511, actual = value + 60 (9 bits)
    flags: int = 0         # 7 bits


# --- GF(2^7) Arithmetic ---

GF_M = 7
GF_SIZE = 1 << GF_M     # 128
GF_MASK = GF_SIZE - 1   # 127
PRIM_POLY = 0b10001001  # x^7 + x^3 + 1 = 137

# Build lookup tables
_gf_exp = np.zeros(GF_SIZE * 2, dtype=np.int32)
_gf_log = np.zeros(GF_SIZE, dtype=np.int32)


def _init_gf():
    x = 1
    for i in range(GF_MASK):
        _gf_exp[i] = x
        _gf_exp[i + GF_MASK] = x
        _gf_log[x] = i
        x <<= 1
        if x & GF_SIZE:
            x ^= PRIM_POLY
    _gf_log[0] = -1


_init_gf()


def _gf_mul(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return int(_gf_exp[int(_gf_log[a]) + int(_gf_log[b])])


# --- BCH Parameters ---

BCH_N = 71
BCH_K = 50
BCH_T = 3
BCH_2T = 6
BCH_R = BCH_N - BCH_K  # 21

_generator_poly: Optional[np.ndarray] = None


def _minimal_poly(i: int) -> list[int]:
    """Compute minimal polynomial of alpha^i over GF(2)."""
    conjugates = []
    exp = i
    for _ in range(GF_M):
        c = exp % GF_MASK
        if c not in conjugates:
            conjugates.append(c)
        exp = (exp * 2) % GF_MASK

    poly = [1]
    for c in conjugates:
        root = int(_gf_exp[c])
        new_poly = [0] * (len(poly) + 1)
        for j in range(len(poly)):
            new_poly[j] ^= poly[j]
            new_poly[j + 1] ^= _gf_mul(poly[j], root)
        poly = new_poly

    return [c & 1 for c in poly]


def _compute_generator() -> np.ndarray:
    """Compute BCH generator polynomial."""
    root_exponents = set()
    for i in range(1, BCH_2T + 1):
        min_conj = i % GF_MASK
        exp = i
        for _ in range(GF_M):
            exp = (exp * 2) % GF_MASK
            if exp < min_conj:
                min_conj = exp
        root_exponents.add(min_conj)

    gen = [1]
    for r in root_exponents:
        mp = _minimal_poly(r)
        result = [0] * (len(gen) + len(mp) - 1)
        for i in range(len(gen)):
            if not gen[i]:
                continue
            for j in range(len(mp)):
                result[i + j] ^= mp[j]
        gen = result

    return np.array(gen, dtype=np.uint8)


def _get_generator() -> np.ndarray:
    global _generator_poly
    if _generator_poly is None:
        _generator_poly = _compute_generator()
    return _generator_poly


def bch_encode(data: np.ndarray) -> np.ndarray:
    """BCH encode: 50 data bits -> 71 codeword bits."""
    assert len(data) == BCH_K, f"BCH encode expects {BCH_K} bits, got {len(data)}"

    gen = _get_generator()
    gen_deg = len(gen) - 1

    # Systematic encoding: compute remainder of (data * x^r) / g(x) over GF(2)
    shifted = np.zeros(BCH_K + gen_deg, dtype=np.uint8)
    shifted[:BCH_K] = data

    for i in range(BCH_K):
        if shifted[i]:
            shifted[i:i + len(gen)] ^= gen

    # Codeword = data || remainder
    codeword = np.zeros(BCH_N, dtype=np.uint8)
    codeword[:BCH_K] = data
    codeword[BCH_K:] = shifted[BCH_K:BCH_K + gen_deg]

    return codeword


# --- Payload packing ---

def _write_bits(bits: np.ndarray, value: int, offset: int, count: int):
    """Write `count` bits of `value` into `bits` array at `offset` (MSB first)."""
    for i in range(count):
        bits[offset + i] = (value >> (count - 1 - i)) & 1


def _read_bits(bits: np.ndarray, offset: int, count: int) -> int:
    """Read `count` bits from `bits` array at `offset` (MSB first)."""
    value = 0
    for i in range(count):
        value = (value << 1) | int(bits[offset + i] & 1)
    return value


def pack_payload(payload: VJSyncPayload) -> np.ndarray:
    """Pack payload into 50-bit array."""
    bits = np.zeros(PAYLOAD_BITS, dtype=np.uint8)
    _write_bits(bits, payload.beat_whole & 0xFFF, *FIELD_BEAT_WHOLE)
    _write_bits(bits, payload.beat_frac & 0xFF, *FIELD_BEAT_FRAC)
    _write_bits(bits, payload.frame_seq & 0x3FFF, *FIELD_FRAME_SEQ)
    _write_bits(bits, payload.bpm_encoded & 0x1FF, *FIELD_BPM)
    _write_bits(bits, payload.flags & 0x7F, *FIELD_FLAGS)
    return bits


def unpack_payload(bits: np.ndarray) -> VJSyncPayload:
    """Unpack 50-bit array into payload."""
    return VJSyncPayload(
        beat_whole=_read_bits(bits, *FIELD_BEAT_WHOLE),
        beat_frac=_read_bits(bits, *FIELD_BEAT_FRAC),
        frame_seq=_read_bits(bits, *FIELD_FRAME_SEQ),
        bpm_encoded=_read_bits(bits, *FIELD_BPM),
        flags=_read_bits(bits, *FIELD_FLAGS),
    )


# --- Encoder ---

def _adaptive_bar_width(width: int) -> int:
    """Compute bar width based on frame width. Minimum 1px per bar."""
    return max(1, width // TOTAL_BARS)


def encode_strip(payload: VJSyncPayload, width: int) -> np.ndarray:
    """
    Encode a VJSync payload into a barcode strip.

    Bar width adapts to frame width:
      - 512px+ → 6px bars (standard)
      - 256px  → 3px bars
      - 85px   → 1px bars (minimum)

    Returns: numpy array of shape (STRIP_HEIGHT, width, 3), dtype uint8
             with ITU-R limited range levels (16=black, 235=white)
    """
    if width < TOTAL_BARS:
        raise ValueError(
            f"Width {width} too narrow for {TOTAL_BARS} bars. "
            f"Need at least {TOTAL_BARS}px ({TOTAL_BARS} bars x 1px)."
        )

    bar_width = _adaptive_bar_width(width)

    # 1. Pack payload -> 50 bits
    data_bits = pack_payload(payload)

    # 2. BCH encode -> 71 bits
    codeword = bch_encode(data_bits)

    # 3. Build bar sequence: sync (14) + data (71) = 85 bars
    bars = np.zeros(TOTAL_BARS, dtype=np.uint8)
    bars[:SYNC_BARS] = SYNC_PATTERN
    bars[SYNC_BARS:] = codeword

    # 4. Render to image
    strip = np.full((STRIP_HEIGHT, width, 3), BLACK_LEVEL, dtype=np.uint8)

    for bar_idx in range(TOTAL_BARS):
        level = WHITE_LEVEL if bars[bar_idx] else BLACK_LEVEL
        x_start = bar_idx * bar_width
        x_end = min(x_start + bar_width, width)
        strip[:, x_start:x_end, :] = level

    return strip


def stamp_barcode(frame: np.ndarray, payload: VJSyncPayload) -> np.ndarray:
    """
    Stamp a VJSync barcode onto the bottom of a frame (in-place).

    Args:
        frame: numpy array (H, W, 3), dtype uint8
        payload: timecode payload to encode

    Returns: the modified frame (same reference)
    """
    H, W, _ = frame.shape
    strip = encode_strip(payload, W)
    frame[-STRIP_HEIGHT:, :, :] = strip
    return frame


# --- Convenience helpers ---

def encode_bpm(bpm: float) -> int:
    """Convert BPM to encoded value (offset 60, range 60-571)."""
    return max(0, min(511, round(bpm - 60)))


def decode_bpm(encoded: int) -> float:
    """Convert encoded value back to BPM."""
    return encoded + 60.0


def encode_beat_frac(frac: float) -> int:
    """Convert fractional beat (0-1) to 8-bit value."""
    return round(max(0.0, min(1.0, frac)) * 255)


def decode_beat_frac(encoded: int) -> float:
    """Convert 8-bit value back to fractional beat (0-1)."""
    return encoded / 255.0


# --- BCH Verification & Decode ---

def bch_verify(received: np.ndarray) -> bool:
    """
    Verify a received codeword by re-encoding the data portion
    and comparing to the received ECC bits.

    This is simpler than syndrome decoding and always correct.
    """
    if len(received) != BCH_N:
        return False
    data = received[:BCH_K]
    expected = bch_encode(data)
    return np.array_equal(received, expected)


def bch_decode(received: np.ndarray) -> tuple[Optional[np.ndarray], int]:
    """
    BCH decode: verify and extract data bits from 71-bit received codeword.

    For clean signals (no bit errors), verifies the ECC matches and returns data.
    For up to BCH_T bit errors, attempts brute-force correction.

    Returns:
        (corrected_data_bits, num_errors) — corrected 50 data bits and error count.
        If uncorrectable, returns (None, -1).
    """
    assert len(received) == BCH_N, f"BCH decode expects {BCH_N} bits, got {len(received)}"

    # Fast path: no errors
    if bch_verify(received):
        return received[:BCH_K].copy(), 0

    # Try single-bit error correction (brute force, fast for n=71)
    for i in range(BCH_N):
        trial = received.copy()
        trial[i] ^= 1
        if bch_verify(trial):
            return trial[:BCH_K].copy(), 1

    # Try double-bit error correction
    for i in range(BCH_N):
        for j in range(i + 1, BCH_N):
            trial = received.copy()
            trial[i] ^= 1
            trial[j] ^= 1
            if bch_verify(trial):
                return trial[:BCH_K].copy(), 2

    # Triple-bit correction is O(n^3) = ~60K iterations, still feasible for n=71
    for i in range(BCH_N):
        for j in range(i + 1, BCH_N):
            for k in range(j + 1, BCH_N):
                trial = received.copy()
                trial[i] ^= 1
                trial[j] ^= 1
                trial[k] ^= 1
                if bch_verify(trial):
                    return trial[:BCH_K].copy(), 3

    return None, -1


# --- Barcode Reader ---

def _read_bar_values(strip: np.ndarray) -> Optional[np.ndarray]:
    """
    Read bar values (0 or 1) from a barcode strip image.

    Args:
        strip: numpy array (H, W, 3), dtype uint8 — the bottom strip of a frame

    Returns:
        Array of TOTAL_BARS bar values (0=black, 1=white), or None if unreadable
    """
    H, W, _ = strip.shape
    if W < TOTAL_BARS:
        return None

    bar_width = _adaptive_bar_width(W)

    # Average across height and channels for robustness
    scanline = strip.mean(axis=(0, 2))  # (W,)

    # Read each bar by sampling center of the bar
    bars = np.zeros(TOTAL_BARS, dtype=np.uint8)
    threshold = (BLACK_LEVEL + WHITE_LEVEL) / 2.0

    for i in range(TOTAL_BARS):
        center = i * bar_width + bar_width // 2
        if center < W:
            bars[i] = 1 if scanline[center] > threshold else 0

    return bars


def _verify_sync(bars: np.ndarray) -> bool:
    """Check if the sync pattern matches."""
    sync = bars[:SYNC_BARS]
    expected = np.array(SYNC_PATTERN, dtype=np.uint8)
    return np.array_equal(sync, expected)


def decode_strip(strip: np.ndarray) -> Optional[VJSyncPayload]:
    """
    Decode a VJSync barcode from a strip image.

    Args:
        strip: numpy array (H, W, 3), dtype uint8 — the barcode strip region

    Returns:
        VJSyncPayload if successfully decoded, None otherwise
    """
    bars = _read_bar_values(strip)
    if bars is None:
        return None

    if not _verify_sync(bars):
        return None

    # Extract data bits (after sync)
    received = bars[SYNC_BARS:]
    assert len(received) == DATA_BARS

    # BCH decode
    corrected, num_errors = bch_decode(received)
    if corrected is None:
        return None

    return unpack_payload(corrected)


def read_barcode(frame: np.ndarray, barcode_height: int = STRIP_HEIGHT) -> Optional[VJSyncPayload]:
    """
    Read a VJSync barcode from the bottom of a frame.

    Args:
        frame: numpy array (H, W, 3), dtype uint8
        barcode_height: height of the barcode strip in pixels

    Returns:
        VJSyncPayload if successfully decoded, None otherwise
    """
    H, W, _ = frame.shape
    strip = frame[-barcode_height:, :, :]
    return decode_strip(strip)
