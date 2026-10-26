"""Synthetic grayscale "part" images with real defect masks (week 25).

Reuses week 05's real, already-tested mask generator
(`src.synthetic.mask_generation.generate_defect_mask`) and its shared
config (`config/defects.yaml`, loaded via
`src.synthetic.config.load_defects_config`) -- the same `scratch`/`pit`/
`crack` shapes week 05's inpainting pipeline draws, not a new defect
vocabulary invented for this week. What's new here is purely the
compositing: week 05 pastes a mask into a diffusion inpainting prompt;
this week darkens the base image directly under the mask, cheaply and
deterministically, because week 25's point is to benchmark classical CV
on PIXELS, not to generate photorealistic training data.

`512x512` matches `scripts/generate_defect_masks.py`'s own default
`--image-size` -- picked there (and reused here) because it is comfortably
larger than `crack`'s worst-case extent at `margin_fraction=0.05`
(`_shape_extent` in `mask_generation.py`), so `generate_defect_mask` never
raises `ValueError: Image size ... too small`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.synthetic.config import DefectsConfig
from src.synthetic.mask_generation import generate_defect_mask

IMAGE_SIZE: tuple[int, int] = (512, 512)
"""`(width, height)` of every generated image -- see module docstring."""

BACKGROUND_INTENSITY = 180
"""Mean grayscale intensity of the clean "part" surface, 0-255."""

BACKGROUND_NOISE_STD = 6.0
"""Gaussian sensor-noise std-dev added to the background, so a flat
threshold isn't trivially perfect -- a real camera frame is never a
perfectly flat gray."""

DEFECT_INTENSITY_DROP = 95
"""How much darker a defect pixel is than the surrounding background --
scratches/pits/cracks read as darker on most metal/plastic parts (the
same assumption `OPERATION-CARDS.md`'s inspection cards make)."""

DEFECT_NOISE_STD = 10.0
"""Gaussian noise added within the defect area -- defects aren't a flat
fill either."""

Box = tuple[float, float, float, float]
"""`(x_min, y_min, x_max, y_max)` pixel bounding box, same convention as
`src.evaluation.detection_metrics.Box`."""


@dataclass(frozen=True)
class GeneratedImage:
    """One synthetic test image plus its ground truth.

    Attributes:
        image_id: Unique identifier (e.g. `"positive_0003"`), matches
            `src.evaluation.detection_metrics.GroundTruth.image_id`.
        pixels: `(height, width)` `uint8` grayscale array.
        ground_truth_boxes: Pixel bounding box of every defect actually
            drawn (empty for a clean/negative image).
        shape_names: `DefectInstance.shape_name` for each box, same order.
    """

    image_id: str
    pixels: np.ndarray
    ground_truth_boxes: tuple[Box, ...]
    shape_names: tuple[str, ...]


def render_background(size: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """Render a clean, noisy-but-flat grayscale "part" surface.

    Args:
        size: `(width, height)`.
        rng: Source of randomness.

    Returns:
        `(height, width)` `uint8` array.
    """
    width, height = size
    noise = rng.normal(loc=0.0, scale=BACKGROUND_NOISE_STD, size=(height, width))
    pixels = np.clip(BACKGROUND_INTENSITY + noise, 0, 255)
    return pixels.astype(np.uint8)


def _clip_box(box: Box, size: tuple[int, int]) -> Box:
    """Clip a possibly out-of-frame box (see `generate_defect_mask`'s docstring) to the canvas."""
    width, height = size
    x_min, y_min, x_max, y_max = box
    return (
        max(0.0, float(x_min)),
        max(0.0, float(y_min)),
        min(float(width), float(x_max)),
        min(float(height), float(y_max)),
    )


def _darken_under_mask(
    background: np.ndarray, mask: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Composite a defect onto `background` wherever `mask` is set.

    Simple alpha-free darken-and-add-noise blend -- deliberately not
    `dataset_pipeline/scripts/01_generate_synthetic.py::overlay_defect_mask`
    (that script-local helper blends a *color* for an RGB diffusion
    prompt; this week stays single-channel and needs a *darker*, noisier
    patch, not a flat color fill), reimplemented fresh the same way
    `src/evaluation/detection_metrics.py`'s mAP math is reimplemented
    fresh from `ros2_ws`'s rather than imported across that workspace
    boundary.
    """
    defect_noise = rng.normal(loc=0.0, scale=DEFECT_NOISE_STD, size=background.shape)
    darkened = np.clip(background.astype(np.float64) - DEFECT_INTENSITY_DROP + defect_noise, 0, 255)
    is_defect = mask > 0
    result = np.where(is_defect, darkened, background.astype(np.float64))
    return result.astype(np.uint8)


def render_defect_image(
    image_id: str, config: DefectsConfig, rng: np.random.Generator
) -> GeneratedImage:
    """Render one image with 1-2 real defects drawn by `generate_defect_mask`.

    Args:
        image_id: Identifier stored on the returned `GeneratedImage`.
        config: Parsed `config/defects.yaml` (shapes/weights/count_range).
        rng: Source of randomness, shared across the whole dataset build
            so results are reproducible for a given seed.

    Returns:
        A `GeneratedImage` with 1-2 ground truth boxes (per
        `config.count_range`).
    """
    background = render_background(IMAGE_SIZE, rng)
    mask, instances = generate_defect_mask(IMAGE_SIZE, config, rng=rng)
    pixels = _darken_under_mask(background, mask, rng)

    boxes = tuple(_clip_box(instance.bbox, IMAGE_SIZE) for instance in instances)
    names = tuple(instance.shape_name for instance in instances)
    return GeneratedImage(
        image_id=image_id, pixels=pixels, ground_truth_boxes=boxes, shape_names=names
    )


def render_clean_image(image_id: str, rng: np.random.Generator) -> GeneratedImage:
    """Render one defect-free image -- a negative, for false-positive measurement.

    Args:
        image_id: Identifier stored on the returned `GeneratedImage`.
        rng: Source of randomness.

    Returns:
        A `GeneratedImage` with empty `ground_truth_boxes`/`shape_names`.
    """
    pixels = render_background(IMAGE_SIZE, rng)
    return GeneratedImage(image_id=image_id, pixels=pixels, ground_truth_boxes=(), shape_names=())


def build_demo_dataset(
    config: DefectsConfig, num_positive: int, num_negative: int, seed: int
) -> list[GeneratedImage]:
    """Build a deterministic positive+negative dataset for the week 25 benchmark.

    Args:
        config: Parsed `config/defects.yaml`.
        num_positive: Number of images with 1-2 real defects each.
        num_negative: Number of clean images.
        seed: RNG seed -- the same seed always reproduces the exact same
            dataset (pixels and ground truth boxes alike).

    Returns:
        `num_positive` `"positive_%04d"` images followed by `num_negative`
        `"negative_%04d"` images.
    """
    rng = np.random.default_rng(seed)
    positives = [render_defect_image(f"positive_{i:04d}", config, rng) for i in range(num_positive)]
    negatives = [render_clean_image(f"negative_{i:04d}", rng) for i in range(num_negative)]
    return positives + negatives
