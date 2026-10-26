"""Tests for `src.scenarios.image_sources` (week 12, day 3)."""

from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from src.scenarios.config import CameraSource
from src.scenarios.image_sources import (
    _camera_image_source,
    _synthesize_colored_image,
    _synthetic_color_image_source,
    _synthetic_texture_image_source,
    build_color_sorting_image_source,
    build_defect_detection_image_source,
)


def test_synthesize_colored_image_is_deterministic_for_the_same_seed() -> None:
    a = _synthesize_colored_image("red", (16, 16), seed=1)
    b = _synthesize_colored_image("red", (16, 16), seed=1)
    assert np.array_equal(np.array(a), np.array(b))


def test_synthesize_colored_image_differs_across_colors() -> None:
    red = _synthesize_colored_image("red", (16, 16), seed=1)
    blue = _synthesize_colored_image("blue", (16, 16), seed=1)
    assert not np.array_equal(np.array(red), np.array(blue))


def test_synthesize_colored_image_rejects_unknown_color() -> None:
    with pytest.raises(KeyError):
        _synthesize_colored_image("purple", (16, 16), seed=1)


def test_synthetic_color_image_source_cycles_and_returns_real_png_bytes() -> None:
    source = _synthetic_color_image_source(("red", "blue"), size=(8, 8))
    frame1 = source()
    frame2 = source()
    assert frame1[:8] == b"\x89PNG\r\n\x1a\n"
    assert Image.open(BytesIO(frame1)).size == (8, 8)
    assert frame1 != frame2  # different color/seed each call


def test_synthetic_texture_image_source_returns_real_png_bytes() -> None:
    source = _synthetic_texture_image_source(size=(8, 8))
    frame = source()
    assert frame[:8] == b"\x89PNG\r\n\x1a\n"
    assert Image.open(BytesIO(frame)).size == (8, 8)


def test_camera_image_source_returns_none_when_source_unreadable() -> None:
    source = _camera_image_source("/dev/definitely-not-a-real-camera-device")
    assert source() is None


def test_build_color_sorting_image_source_synthetic() -> None:
    source = build_color_sorting_image_source(CameraSource("synthetic"), ("red", "green"))
    frame = source()
    assert frame is not None
    assert Image.open(BytesIO(frame)).mode == "RGB"


def test_build_color_sorting_image_source_real_camera_path() -> None:
    source = build_color_sorting_image_source(
        CameraSource("/dev/definitely-not-a-real-camera-device"), ("red",)
    )
    assert source() is None


def test_build_defect_detection_image_source_synthetic() -> None:
    source = build_defect_detection_image_source(CameraSource("synthetic"))
    frame = source()
    assert frame is not None


def test_build_defect_detection_image_source_real_camera_path() -> None:
    source = build_defect_detection_image_source(
        CameraSource("/dev/definitely-not-a-real-camera-device")
    )
    assert source() is None
