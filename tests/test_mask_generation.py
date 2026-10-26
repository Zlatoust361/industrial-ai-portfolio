"""Unit tests for `src.synthetic.mask_generation`."""

from __future__ import annotations

import numpy as np
import pytest

from src.synthetic.config import DefectsConfig, ShapeSpec
from src.synthetic.mask_generation import (
    choose_shape,
    generate_defect_mask,
    mask_coverage_fraction,
    sample_position,
)

LINE_SHAPE = ShapeSpec(
    name="scratch", type="line", weight=1.0, width_range=(2, 4), length_range=(20, 30)
)
ELLIPSE_SHAPE = ShapeSpec(name="pit", type="ellipse", weight=1.0, size_range=(10, 20))
POLYLINE_SHAPE = ShapeSpec(
    name="crack",
    type="polyline",
    weight=1.0,
    width_range=(1, 2),
    length_range=(30, 40),
    segments_range=(3, 4),
)


def _config(
    shapes: tuple[ShapeSpec, ...], count_range: tuple[int, int] = (1, 1), seed: int = 0
) -> DefectsConfig:
    return DefectsConfig(seed=seed, margin_fraction=0.05, count_range=count_range, shapes=shapes)


# -- choose_shape --------------------------------------------------------


def test_choose_shape_empty_raises() -> None:
    with pytest.raises(ValueError, match="No shapes"):
        choose_shape((), np.random.default_rng(0))


def test_choose_shape_all_zero_weight_raises() -> None:
    shapes = (ShapeSpec(name="a", type="ellipse", weight=0.0, size_range=(1, 2)),)
    with pytest.raises(ValueError, match="positive"):
        choose_shape(shapes, np.random.default_rng(0))


def test_choose_shape_never_picks_a_zero_weight_shape() -> None:
    shapes = (
        ShapeSpec(name="never", type="ellipse", weight=0.0, size_range=(1, 2)),
        ShapeSpec(name="always", type="ellipse", weight=1.0, size_range=(1, 2)),
    )
    rng = np.random.default_rng(0)
    picks = {choose_shape(shapes, rng).name for _ in range(200)}
    assert picks == {"always"}


# -- sample_position ------------------------------------------------------


def test_sample_position_stays_within_margin_bounds() -> None:
    rng = np.random.default_rng(0)
    image_size = (100, 100)
    extent = 20
    margin_fraction = 0.1
    margin = int(100 * margin_fraction)  # 10
    half = extent // 2  # 10

    for _ in range(200):
        x, y = sample_position(image_size, extent, margin_fraction, rng)
        assert margin + half <= x <= 100 - margin - half
        assert margin + half <= y <= 100 - margin - half


def test_sample_position_too_small_for_extent_raises() -> None:
    with pytest.raises(ValueError, match="too small"):
        sample_position((20, 20), extent=50, margin_fraction=0.05, rng=np.random.default_rng(0))


# -- generate_defect_mask --------------------------------------------------


def test_generate_defect_mask_is_reproducible_with_the_same_config_seed() -> None:
    config = _config((LINE_SHAPE, ELLIPSE_SHAPE, POLYLINE_SHAPE), count_range=(2, 2), seed=42)

    mask_a, instances_a = generate_defect_mask((256, 256), config)
    mask_b, instances_b = generate_defect_mask((256, 256), config)

    np.testing.assert_array_equal(mask_a, mask_b)
    assert instances_a == instances_b


def test_generate_defect_mask_different_seeds_produce_different_masks() -> None:
    config_a = _config((LINE_SHAPE, ELLIPSE_SHAPE, POLYLINE_SHAPE), count_range=(2, 2), seed=1)
    config_b = _config((LINE_SHAPE, ELLIPSE_SHAPE, POLYLINE_SHAPE), count_range=(2, 2), seed=2)

    mask_a, _ = generate_defect_mask((256, 256), config_a)
    mask_b, _ = generate_defect_mask((256, 256), config_b)

    assert not np.array_equal(mask_a, mask_b)


def test_generate_defect_mask_shape_of_output() -> None:
    config = _config((ELLIPSE_SHAPE,))
    mask, _ = generate_defect_mask((300, 200), config)
    assert mask.shape == (200, 300)  # (height, width), numpy convention
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 255})


@pytest.mark.parametrize("fixed_count", [1, 2, 3])
def test_generate_defect_mask_respects_fixed_count_range(fixed_count: int) -> None:
    config = _config((ELLIPSE_SHAPE,), count_range=(fixed_count, fixed_count))
    _, instances = generate_defect_mask((512, 512), config)
    assert len(instances) == fixed_count


def test_generate_defect_mask_nonzero_pixels_stay_within_reported_bboxes() -> None:
    config = _config((LINE_SHAPE, ELLIPSE_SHAPE, POLYLINE_SHAPE), count_range=(3, 3), seed=7)
    image_size = (400, 400)
    mask, instances = generate_defect_mask(image_size, config)

    allowed = np.zeros(mask.shape, dtype=bool)
    for instance in instances:
        x_min, y_min, x_max, y_max = instance.bbox
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(image_size[0], x_max), min(image_size[1], y_max)
        allowed[y_min:y_max, x_min:x_max] = True

    assert np.all(allowed | (mask == 0))


def test_generate_defect_mask_polyline_bbox_stays_within_image_bounds() -> None:
    # Regression test: _draw_polyline's random walk starts AT the sampled
    # center and can wander up to its full length in one direction (not
    # symmetric like _draw_line), so a _shape_extent that budgets margin
    # as if it were symmetric under-estimates the required clearance and
    # produces boxes that spill outside the image -- caught by week 06's
    # dataset validator (dataset_pipeline/scripts/08_validate_dataset.py)
    # against the real config/defects.yaml crack shape.
    config = _config((POLYLINE_SHAPE,), count_range=(1, 1))
    image_size = (200, 200)

    for seed in range(300):
        rng = np.random.default_rng(seed)
        _, instances = generate_defect_mask(image_size, config, rng)
        for instance in instances:
            x_min, y_min, x_max, y_max = instance.bbox
            assert 0 <= x_min < x_max <= image_size[0]
            assert 0 <= y_min < y_max <= image_size[1]


def test_generate_defect_mask_image_too_small_raises() -> None:
    config = _config((ELLIPSE_SHAPE,))  # size_range up to 20px
    with pytest.raises(ValueError, match="too small"):
        generate_defect_mask((5, 5), config)


def test_generate_defect_mask_without_rng_uses_config_seed_deterministically() -> None:
    config = _config((LINE_SHAPE, ELLIPSE_SHAPE), count_range=(2, 2), seed=99)

    mask_a, _ = generate_defect_mask((256, 256), config)  # rng=None both times
    mask_b, _ = generate_defect_mask((256, 256), config)

    np.testing.assert_array_equal(mask_a, mask_b)


# -- mask_coverage_fraction -------------------------------------------------


def test_mask_coverage_fraction_all_defect() -> None:
    assert mask_coverage_fraction(np.full((10, 10), 255, dtype=np.uint8)) == 1.0


def test_mask_coverage_fraction_no_defect() -> None:
    assert mask_coverage_fraction(np.zeros((10, 10), dtype=np.uint8)) == 0.0


def test_mask_coverage_fraction_partial() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[0:5, :] = 255
    assert mask_coverage_fraction(mask) == pytest.approx(0.5)


def test_mask_coverage_fraction_empty_mask_is_zero() -> None:
    assert mask_coverage_fraction(np.zeros((0, 0), dtype=np.uint8)) == 0.0
