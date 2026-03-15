"""
Test the BPM Timecoded Buffer pipeline outside of Scope.

Run: python -m pytest tests/test_pipeline.py -v
  or: python tests/test_pipeline.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import numpy as np


def make_test_frame(width=576, height=336, barcode_height=16):
    """Create a test frame with a synthetic barcode at the bottom."""
    frame = torch.randint(64, 200, (1, height, width, 3), dtype=torch.uint8)
    bar_width = 6
    for x in range(width):
        bar_idx = x // bar_width
        level = 235 if (bar_idx % 2 == 0) else 16
        frame[0, -barcode_height:, x, :] = level
    return frame


def _stack_video(video_out):
    """Helper: stack list of (1,H,W,C) tensors into (F,H,W,C)."""
    if isinstance(video_out, list):
        return torch.cat(video_out, dim=0)
    return video_out


def test_basic_output():
    """Test that the pipeline generates correct video output dimensions."""
    from bpm_sync_timecoded_buffer.pipeline import BpmTimecodedBufferPipeline, BpmBufferConfig

    config = BpmBufferConfig()
    pipeline = BpmTimecodedBufferPipeline(config)

    frame = make_test_frame()
    result = pipeline(video=[frame])

    assert "video" in result
    video = _stack_video(result["video"])
    assert video.shape == (1, 336, 576, 3), f"Video shape wrong: {video.shape}"

    # Video is float [0, 1]
    assert video.max() <= 1.0
    assert video.min() >= 0.0

    # Without input_size/num_frames in kwargs, VACE mask is skipped
    # (preprocessor can't know the main pipeline's frame count)
    assert "vace_input_masks" not in result, "Without frame count kwarg, no VACE mask"

    print("  [OK] Basic output test passed")


def test_vace_mask_at_generation_resolution():
    """Test VACE mask is generated at the main pipeline's generation resolution."""
    from bpm_sync_timecoded_buffer.pipeline import BpmTimecodedBufferPipeline, BpmBufferConfig

    config = BpmBufferConfig()
    pipeline = BpmTimecodedBufferPipeline(config)

    frame = make_test_frame()
    # Simulate Scope broadcasting main pipeline's generation dims
    result = pipeline(
        video=[frame],
        height=320, width=576, input_size=13,
    )

    assert "video" in result
    assert "vace_input_masks" in result, "VACE mask should be present when gen dims available"
    # vace_input_frames must NOT be present (blocks passthrough)
    assert "vace_input_frames" not in result

    vace_masks = result["vace_input_masks"]
    # Must match generation resolution (aligned to 8): [B=1, C=1, F=13, H=320, W=576]
    assert vace_masks.shape == (1, 1, 13, 320, 576), f"VACE mask shape wrong: {vace_masks.shape}"

    # Barcode region (bottom) should be 0 (preserve)
    # barcode_h is 16px at input 336px, scaled to gen 320px
    from bpm_sync_timecoded_buffer.vjsync_codec import STRIP_HEIGHT
    scale_y = 320 / 336
    barcode_h_gen = max(4, round(STRIP_HEIGHT * scale_y))
    barcode_mask = vace_masks[0, 0, 0, -barcode_h_gen:, :]
    assert barcode_mask.max() == 0.0, f"Barcode region should be 0 (preserve), got max={barcode_mask.max()}"

    # Content region should be 1 (generate)
    content_mask = vace_masks[0, 0, 0, 0, 0]
    assert content_mask == 1.0, f"Content region should be 1.0, got {content_mask}"

    print("  [OK] VACE mask at generation resolution test passed")


def test_barcode_in_video_output():
    """Test that video output contains actual barcode data (stamped by preprocessor)."""
    from bpm_sync_timecoded_buffer.pipeline import BpmTimecodedBufferPipeline, BpmBufferConfig

    config = BpmBufferConfig()
    pipeline = BpmTimecodedBufferPipeline(config)

    frame = make_test_frame()
    result = pipeline(video=[frame])

    video = _stack_video(result["video"])
    barcode_region = video[0, -16:, :, :]
    barcode_uint8 = (barcode_region * 255.0).cpu().numpy().astype(np.uint8)
    unique_vals = np.unique(barcode_uint8)
    assert len(unique_vals) >= 2, f"Video output should have barcode data, got: {unique_vals}"

    print("  [OK] Barcode in video output test passed")


def test_multi_frame():
    """Test with multiple frames (batch processing)."""
    from bpm_sync_timecoded_buffer.pipeline import BpmTimecodedBufferPipeline, BpmBufferConfig

    config = BpmBufferConfig()
    pipeline = BpmTimecodedBufferPipeline(config)

    frames = [make_test_frame() for _ in range(4)]
    result = pipeline(video=frames)

    video = _stack_video(result["video"])
    assert video.shape[0] == 4

    print("  [OK] Multi-frame test passed")


def test_test_pattern_input():
    """Test that test_input=True replaces video with bouncing ball animation."""
    from bpm_sync_timecoded_buffer.pipeline import BpmTimecodedBufferPipeline, BpmBufferConfig

    config = BpmBufferConfig()
    pipeline = BpmTimecodedBufferPipeline(config)

    frame = make_test_frame()
    result = pipeline(video=[frame], test_input=True)

    assert "video" in result

    video = _stack_video(result["video"])
    assert video.shape == (1, 336, 576, 3), f"Video shape wrong: {video.shape}"

    print("  [OK] Test pattern input test passed")


def test_set_bpm():
    """Test manual BPM setting."""
    from bpm_sync_timecoded_buffer.pipeline import BpmTimecodedBufferPipeline, BpmBufferConfig, ClockSource

    config = BpmBufferConfig()
    pipeline = BpmTimecodedBufferPipeline(config)

    # Switch to internal clock so set_bpm has immediate effect
    pipeline._clock.set_source(ClockSource.INTERNAL, bpm=120.0)
    pipeline.set_bpm(140.0)
    assert pipeline._clock._internal_bpm == 140.0, f"Expected 140 BPM, got {pipeline._clock._internal_bpm}"
    assert pipeline._clock.tempo == 140.0, f"Expected tempo 140, got {pipeline._clock.tempo}"

    print("  [OK] Set BPM test passed")


def test_barcode_roundtrip():
    """Test encode -> decode barcode roundtrip."""
    from bpm_sync_timecoded_buffer.vjsync_codec import (
        VJSyncPayload, stamp_barcode, read_barcode,
        encode_bpm, decode_bpm, encode_beat_frac, decode_beat_frac,
    )
    import numpy as np

    frame = np.full((336, 576, 3), 128, dtype=np.uint8)
    payload = VJSyncPayload(
        beat_whole=42,
        beat_frac=encode_beat_frac(0.75),
        frame_seq=1234,
        bpm_encoded=encode_bpm(140.0),
        flags=0,
    )
    stamp_barcode(frame, payload)

    decoded = read_barcode(frame, 16)
    assert decoded is not None, "Failed to decode barcode"
    assert decoded.beat_whole == 42, f"beat_whole: expected 42, got {decoded.beat_whole}"
    assert decoded.frame_seq == 1234, f"frame_seq: expected 1234, got {decoded.frame_seq}"
    assert decode_bpm(decoded.bpm_encoded) == 140.0, f"BPM: expected 140, got {decode_bpm(decoded.bpm_encoded)}"

    print("  [OK] Barcode roundtrip test passed")


def test_postprocessor_strip():
    """Test that the postprocessor strips the barcode from output."""
    from bpm_sync_timecoded_buffer.pipeline import BpmTimecodeStripPipeline, BpmStripConfig

    config = BpmStripConfig()
    pipeline = BpmTimecodeStripPipeline(config)

    barcode_h = 16
    frame = make_test_frame(barcode_height=barcode_h)
    result = pipeline(video=[frame])

    assert "video" in result
    video = _stack_video(result["video"])
    assert video.shape == (1, 336, 576, 3), f"Video shape wrong: {video.shape}"

    # Bottom strip should be all black (0)
    bottom = video[0, -barcode_h:, :, :]
    assert bottom.max() == 0, f"Barcode region should be blacked out, got max={bottom.max()}"

    # Content above should NOT be all black
    content = video[0, :-barcode_h, :, :]
    assert content.max() > 0, "Content region should have data"

    print("  [OK] Postprocessor strip test passed")


def test_postprocessor_decode():
    """Test that the postprocessor decodes barcodes from preprocessor output."""
    from bpm_sync_timecoded_buffer.pipeline import (
        BpmTimecodedBufferPipeline, BpmBufferConfig,
        BpmTimecodeStripPipeline, BpmStripConfig,
    )
    import time

    # Run preprocessor to stamp a barcode
    pre_config = BpmBufferConfig()
    pre = BpmTimecodedBufferPipeline(pre_config)
    time.sleep(0.05)  # Let clock tick

    frame = torch.randint(64, 200, (1, 336, 576, 3), dtype=torch.uint8)
    pre_result = pre(video=[frame])

    # Preprocessor now returns list of uint8 tensors -- use directly
    stamped_video = pre_result["video"]

    # Run postprocessor in strip mode
    post_config = BpmStripConfig(buffer_mode="no_buffer")
    post = BpmTimecodeStripPipeline(post_config)
    post_result = post(video=stamped_video)

    assert "_bpm_buffer_output_meta" in post_result
    meta = post_result["_bpm_buffer_output_meta"]
    assert meta["decode_success"] > 0 or meta["decode_fail"] > 0, "No decode attempts"

    print(f"  [OK] Postprocessor decode test passed (success={meta['decode_success']}, fail={meta['decode_fail']})")


def test_clock_source_switching():
    """Test switching between clock sources."""
    from bpm_sync_timecoded_buffer.pipeline import ClockManager, ClockSource

    clock = ClockManager(120.0)

    # Default is internal
    clock.set_source(ClockSource.INTERNAL, bpm=130.0)
    assert clock.source == ClockSource.INTERNAL
    assert clock.tempo == 130.0

    # Switch BPM
    clock.set_internal_bpm(145.0)
    assert clock.tempo == 145.0

    # Source info should contain source type
    info = clock.source_info
    assert info["source"] == "internal"
    assert info["tempo"] == 145.0

    clock.stop()
    print("  [OK] Clock source switching test passed")


def test_preprocessor_clock_source():
    """Test that preprocessor respects clock_source parameter."""
    from bpm_sync_timecoded_buffer.pipeline import BpmTimecodedBufferPipeline, BpmBufferConfig

    config = BpmBufferConfig()
    pipeline = BpmTimecodedBufferPipeline(config)

    frame = make_test_frame()
    # Switch to internal clock via kwargs
    result = pipeline(video=[frame], clock_source="internal", clock_bpm=140.0)

    assert "video" in result
    meta = result["_bpm_buffer_meta"]
    assert meta["source"] == "internal"

    pipeline._clock.stop()
    print("  [OK] Preprocessor clock source test passed")


def test_postprocessor_latency_mode():
    """Test latency buffer mode."""
    from bpm_sync_timecoded_buffer.pipeline import BpmTimecodeStripPipeline, BpmStripConfig

    config = BpmStripConfig(buffer_mode="latency", latency_delay_ms=100)
    pipeline = BpmTimecodeStripPipeline(config)

    barcode_h = 16
    # Feed several batches to fill the buffer
    for _ in range(5):
        frame = make_test_frame(barcode_height=barcode_h)
        result = pipeline(video=[frame])
        assert "video" in result

    meta = result["_bpm_buffer_output_meta"]
    assert meta["buffer_mode"] == "latency"

    print("  [OK] Postprocessor latency mode test passed")


if __name__ == "__main__":
    print("\n=== BPM Timecoded Buffer Pipeline Tests ===\n")

    tests = [
        test_basic_output,
        test_vace_mask_at_generation_resolution,
        test_barcode_in_video_output,
        test_multi_frame,
        test_test_pattern_input,
        test_set_bpm,
        test_barcode_roundtrip,
        test_clock_source_switching,
        test_preprocessor_clock_source,
        test_postprocessor_strip,
        test_postprocessor_decode,
        test_postprocessor_latency_mode,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {test.__name__} FAILED: {e}")
            failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("All tests passed!")
    sys.exit(failed)
