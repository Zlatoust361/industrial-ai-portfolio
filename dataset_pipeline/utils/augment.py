"""Bbox-safe image augmentations (week 06, stage 3).

One geometric op (horizontal flip -- boxes are flipped to match exactly)
and three photometric ops (brightness/contrast jitter, light gaussian
noise -- none of which move a pixel, so boxes are untouched). Deliberately
not rotation/crop: those need real bbox-clipping math to stay correct,
and this stage's job is "add training diversity", not a full augmentation
library. Pure PIL/numpy, no torch.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageEnhance

from utils.manifest import Bbox

AUGMENTATIONS: tuple[str, ...] = ("hflip", "brightness_jitter", "contrast_jitter", "gaussian_noise")


def hflip_image_and_boxes(
    image: Image.Image, boxes: list[Bbox]
) -> tuple[Image.Image, list[Bbox]]:
    """Mirror `image` horizontally, flipping every box's x-coordinates to match.

    Args:
        image: Source image.
        boxes: Boxes in `image`'s coordinate frame.

    Returns:
        `(flipped_image, flipped_boxes)`, same order as `boxes`.
    """
    flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
    width = image.width
    flipped_boxes = [(width - x1, y0, width - x0, y1) for (x0, y0, x1, y1) in boxes]
    return flipped, flipped_boxes


def jitter_brightness(image: Image.Image, factor: float) -> Image.Image:
    """Multiply `image`'s brightness by `factor` (bbox-unaffected)."""
    return ImageEnhance.Brightness(image).enhance(factor)


def jitter_contrast(image: Image.Image, factor: float) -> Image.Image:
    """Multiply `image`'s contrast by `factor` (bbox-unaffected)."""
    return ImageEnhance.Contrast(image).enhance(factor)


def add_gaussian_noise(image: Image.Image, std: float, rng: np.random.Generator) -> Image.Image:
    """Add zero-mean gaussian noise (bbox-unaffected).

    Args:
        image: Source image.
        std: Noise standard deviation, as a fraction of the `[0, 255]` range.
        rng: Source of randomness.

    Returns:
        A new image, pixel values clipped back to `[0, 255]`.
    """
    array = np.asarray(image, dtype=np.float32)
    noisy = array + rng.normal(0.0, std * 255.0, size=array.shape)
    return Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8))


def apply_augmentation(
    image: Image.Image,
    object_bbox: Bbox,
    defect_boxes: list[Bbox],
    augmentation: str,
    rng: np.random.Generator,
) -> tuple[Image.Image, Bbox, list[Bbox]]:
    """Apply one named augmentation, returning the transformed image and boxes.

    Args:
        image: Source image.
        object_bbox: The sample's object bounding box.
        defect_boxes: The sample's defect bounding boxes (possibly empty).
        augmentation: One of `AUGMENTATIONS`.
        rng: Source of randomness for jitter strength/noise.

    Returns:
        `(augmented_image, augmented_object_bbox, augmented_defect_boxes)`.

    Raises:
        ValueError: If `augmentation` isn't one of `AUGMENTATIONS`.
    """
    if augmentation == "hflip":
        flipped, boxes = hflip_image_and_boxes(image, [object_bbox, *defect_boxes])
        return flipped, boxes[0], boxes[1:]
    if augmentation == "brightness_jitter":
        return jitter_brightness(image, float(rng.uniform(0.7, 1.3))), object_bbox, defect_boxes
    if augmentation == "contrast_jitter":
        return jitter_contrast(image, float(rng.uniform(0.8, 1.2))), object_bbox, defect_boxes
    if augmentation == "gaussian_noise":
        return add_gaussian_noise(image, 0.02, rng), object_bbox, defect_boxes
    raise ValueError(f"Unknown augmentation {augmentation!r}; expected one of {AUGMENTATIONS}.")
