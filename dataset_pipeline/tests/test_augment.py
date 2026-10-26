"""Unit tests for `utils.augment` -- real PIL/numpy, no mocking."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image
from utils.augment import (
    add_gaussian_noise,
    apply_augmentation,
    hflip_image_and_boxes,
    jitter_brightness,
    jitter_contrast,
)


def _gradient_image(width: int = 20, height: int = 10) -> Image.Image:
    array = np.zeros((height, width, 3), dtype=np.uint8)
    array[:, :, 0] = np.linspace(0, 255, width, dtype=np.uint8)
    return Image.fromarray(array)


def test_hflip_image_and_boxes_mirrors_pixels_and_boxes() -> None:
    image = _gradient_image(width=20, height=10)
    boxes = [(2, 2, 8, 8)]

    flipped, flipped_boxes = hflip_image_and_boxes(image, boxes)

    assert flipped.size == image.size
    assert flipped_boxes == [(12, 2, 18, 8)]  # width=20 -> 20-8=12, 20-2=18
    # leftmost column of the gradient (value ~0) should now be on the right.
    assert np.array(flipped)[0, -1, 0] < np.array(flipped)[0, 0, 0]


def test_jitter_brightness_changes_pixel_values() -> None:
    image = Image.new("RGB", (10, 10), color=(100, 100, 100))

    brighter = jitter_brightness(image, 1.5)

    assert np.array(brighter)[0, 0, 0] > 100


def test_jitter_contrast_changes_pixel_values() -> None:
    image = _gradient_image()

    contrasted = jitter_contrast(image, 1.5)

    assert not np.array_equal(np.array(contrasted), np.array(image))


def test_add_gaussian_noise_changes_pixels_but_stays_valid_image() -> None:
    image = Image.new("RGB", (10, 10), color=(128, 128, 128))
    rng = np.random.default_rng(0)

    noisy = add_gaussian_noise(image, std=0.05, rng=rng)

    array = np.array(noisy)
    assert array.dtype == np.uint8
    assert not np.array_equal(array, np.array(image))


def test_apply_augmentation_hflip_moves_boxes() -> None:
    image = _gradient_image(width=20, height=10)
    rng = np.random.default_rng(0)

    augmented_image, object_bbox, defect_boxes = apply_augmentation(
        image, (0, 0, 10, 10), [(2, 2, 4, 4)], "hflip", rng
    )

    assert augmented_image.size == image.size
    assert object_bbox == (10, 0, 20, 10)
    assert defect_boxes == [(16, 2, 18, 4)]


def test_apply_augmentation_photometric_ops_leave_boxes_untouched() -> None:
    image = _gradient_image()
    rng = np.random.default_rng(0)

    for augmentation in ("brightness_jitter", "contrast_jitter", "gaussian_noise"):
        _, object_bbox, defect_boxes = apply_augmentation(
            image, (0, 0, 5, 5), [(1, 1, 2, 2)], augmentation, rng
        )
        assert object_bbox == (0, 0, 5, 5)
        assert defect_boxes == [(1, 1, 2, 2)]


def test_apply_augmentation_rejects_unknown_name() -> None:
    rng = np.random.default_rng(0)

    with pytest.raises(ValueError, match="Unknown augmentation"):
        apply_augmentation(_gradient_image(), (0, 0, 5, 5), [], "rotate90", rng)
