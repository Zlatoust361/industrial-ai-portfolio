"""Unit tests for `classical_cv.defect_images`."""

from __future__ import annotations

import numpy as np

from classical_cv.defect_images import (
    IMAGE_SIZE,
    build_demo_dataset,
    render_background,
    render_clean_image,
    render_defect_image,
)
from src.synthetic.config import load_defects_config

DEFECTS_CONFIG = load_defects_config("config/defects.yaml")


def test_render_background_shape_and_dtype() -> None:
    rng = np.random.default_rng(0)
    background = render_background(IMAGE_SIZE, rng)
    width, height = IMAGE_SIZE
    assert background.shape == (height, width)
    assert background.dtype == np.uint8


def test_render_defect_image_has_at_least_one_ground_truth_box() -> None:
    rng = np.random.default_rng(1)
    image = render_defect_image("test_0000", DEFECTS_CONFIG, rng)
    assert 1 <= len(image.ground_truth_boxes) <= 2  # config/defects.yaml's count_range
    assert len(image.shape_names) == len(image.ground_truth_boxes)
    for name in image.shape_names:
        assert name in {"scratch", "pit", "crack"}


def test_render_defect_image_boxes_are_within_canvas() -> None:
    rng = np.random.default_rng(2)
    width, height = IMAGE_SIZE
    image = render_defect_image("test_0001", DEFECTS_CONFIG, rng)
    for x_min, y_min, x_max, y_max in image.ground_truth_boxes:
        assert 0.0 <= x_min < x_max <= width
        assert 0.0 <= y_min < y_max <= height


def test_render_defect_image_actually_darkens_pixels_under_the_mask() -> None:
    """At least some pixels inside the ground truth box must read measurably
    darker than the clean background -- otherwise classical CV has nothing
    real to detect and the whole week 25 benchmark would be testing noise,
    not defects. Uses the patch's MINIMUM, not its mean: a thin diagonal
    `scratch`/`crack` line only darkens a small fraction of its own
    (axis-aligned) bounding box, so the patch average stays close to
    background even though the line itself is real and dark."""
    rng = np.random.default_rng(3)
    image = render_defect_image("test_0002", DEFECTS_CONFIG, rng)
    x_min, y_min, x_max, y_max = (int(v) for v in image.ground_truth_boxes[0])
    patch = image.pixels[y_min:y_max, x_min:x_max]
    assert patch.size > 0
    assert float(patch.min()) < float(image.pixels.mean()) - 40.0


def test_render_clean_image_has_no_ground_truth() -> None:
    rng = np.random.default_rng(4)
    image = render_clean_image("negative_0000", rng)
    assert image.ground_truth_boxes == ()
    assert image.shape_names == ()


def test_build_demo_dataset_counts_and_ids() -> None:
    dataset = build_demo_dataset(DEFECTS_CONFIG, num_positive=3, num_negative=2, seed=42)
    assert len(dataset) == 5
    positive_ids = [img.image_id for img in dataset if img.ground_truth_boxes]
    negative_ids = [img.image_id for img in dataset if not img.ground_truth_boxes]
    assert positive_ids == ["positive_0000", "positive_0001", "positive_0002"]
    assert negative_ids == ["negative_0000", "negative_0001"]


def test_build_demo_dataset_is_deterministic_for_the_same_seed() -> None:
    first = build_demo_dataset(DEFECTS_CONFIG, num_positive=2, num_negative=2, seed=7)
    second = build_demo_dataset(DEFECTS_CONFIG, num_positive=2, num_negative=2, seed=7)
    for image_a, image_b in zip(first, second, strict=True):
        np.testing.assert_array_equal(image_a.pixels, image_b.pixels)
        assert image_a.ground_truth_boxes == image_b.ground_truth_boxes


def test_build_demo_dataset_differs_across_seeds() -> None:
    first = build_demo_dataset(DEFECTS_CONFIG, num_positive=2, num_negative=0, seed=1)
    second = build_demo_dataset(DEFECTS_CONFIG, num_positive=2, num_negative=0, seed=2)
    assert not np.array_equal(first[0].pixels, second[0].pixels)
