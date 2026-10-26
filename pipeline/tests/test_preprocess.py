"""Unit tests for utils/preprocess.py.

Run with (from pipeline/):
    PYTHONPATH=. pytest tests/test_preprocess.py
"""

import pytest
from utils.preprocess import compute_letterbox_geometry


def test_letterbox_same_aspect_ratio_fills_canvas_exactly() -> None:
    geometry = compute_letterbox_geometry(original_size=(100, 100), target_size=(224, 224))

    assert geometry.resized_size == (224, 224)
    assert geometry.offset == (0, 0)
    assert geometry.scale == pytest.approx(2.24)


def test_letterbox_wide_image_pads_top_and_bottom() -> None:
    # 400x100 into a 200x200 canvas -> scale limited by width (200/400=0.5) -> 200x50, centered.
    geometry = compute_letterbox_geometry(original_size=(400, 100), target_size=(200, 200))

    assert geometry.resized_size == (200, 50)
    assert geometry.offset == (0, 75)
    assert geometry.scale == pytest.approx(0.5)


def test_letterbox_tall_image_pads_left_and_right() -> None:
    # 100x400 into a 200x200 canvas -> scale limited by height (200/400=0.5) -> 50x200, centered.
    geometry = compute_letterbox_geometry(original_size=(100, 400), target_size=(200, 200))

    assert geometry.resized_size == (50, 200)
    assert geometry.offset == (75, 0)
    assert geometry.scale == pytest.approx(0.5)


def test_letterbox_upscales_small_image() -> None:
    geometry = compute_letterbox_geometry(original_size=(50, 50), target_size=(200, 200))

    assert geometry.resized_size == (200, 200)
    assert geometry.scale == pytest.approx(4.0)


@pytest.mark.parametrize(
    ("original_size", "target_size"),
    [
        ((0, 100), (200, 200)),
        ((100, 0), (200, 200)),
        ((100, 100), (0, 200)),
        ((100, 100), (200, -1)),
    ],
)
def test_letterbox_rejects_non_positive_dimensions(original_size, target_size) -> None:
    with pytest.raises(ValueError, match="positive"):
        compute_letterbox_geometry(original_size, target_size)
