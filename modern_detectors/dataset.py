"""Shared dataset prep for week 26: resize week 25's real synthetic defects
for RT-DETR/YOLO training, and write a YOLO-format directory on disk.

Reuses `classical_cv.defect_images.build_demo_dataset` as-is (week 25's
real generator, itself built on week 05's real `generate_defect_mask`) --
week 26 does NOT invent a new defect dataset, it resizes the same one so
the three approaches compared in this week (classical CV, RT-DETR, YOLO)
are measured on the same underlying data, at whatever resolution each
training loop actually needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from classical_cv.defect_images import Box, GeneratedImage

TRAIN_IMAGE_SIZE = 224
"""Square resolution both RT-DETR and YOLO train/eval at in this week.
Divisible by YOLO's stride-32 requirement; small enough that a from-
scratch tiny RT-DETR step takes well under a second on CPU (measured:
~0.19s/step at batch=4, see `modern_detectors/README.md`)."""

DEFECT_CLASS_NAME = "defect"
DEFECT_CLASS_INDEX = 0


@dataclass(frozen=True)
class ResizedImage:
    """One `GeneratedImage`, resized to `TRAIN_IMAGE_SIZE` with its boxes rescaled to match.

    Attributes:
        image_id: Same as the source `GeneratedImage.image_id`.
        rgb_pixels: `(TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE, 3)` `uint8` array
            (grayscale replicated across channels -- the source images
            have no color information to begin with).
        ground_truth_boxes: Boxes rescaled into `TRAIN_IMAGE_SIZE` pixel space.
    """

    image_id: str
    rgb_pixels: np.ndarray
    ground_truth_boxes: tuple[Box, ...]


def resize_image(image: GeneratedImage, source_size: tuple[int, int]) -> ResizedImage:
    """Resize one `GeneratedImage` (and its boxes) to `TRAIN_IMAGE_SIZE`.

    Args:
        image: A week 25 `GeneratedImage` (grayscale, `source_size` pixels).
        source_size: `(width, height)` of `image.pixels` (e.g.
            `classical_cv.defect_images.IMAGE_SIZE`).

    Returns:
        A `ResizedImage` at `(TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE)`.
    """
    source_width, source_height = source_size
    scale_x = TRAIN_IMAGE_SIZE / source_width
    scale_y = TRAIN_IMAGE_SIZE / source_height

    pil_image = Image.fromarray(image.pixels, mode="L").resize(
        (TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE), resample=Image.BILINEAR
    )
    rgb_pixels = np.asarray(pil_image.convert("RGB"))

    rescaled_boxes = tuple(
        (x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y)
        for x0, y0, x1, y1 in image.ground_truth_boxes
    )
    return ResizedImage(
        image_id=image.image_id, rgb_pixels=rgb_pixels, ground_truth_boxes=rescaled_boxes
    )


def resize_dataset(
    images: list[GeneratedImage], source_size: tuple[int, int]
) -> list[ResizedImage]:
    """`resize_image` applied to a whole dataset, same order."""
    return [resize_image(image, source_size) for image in images]


def split_train_val(
    images: list[ResizedImage], val_fraction: float
) -> tuple[list[ResizedImage], list[ResizedImage]]:
    """Stratified train/val split -- keeps the same positive/negative ratio in both halves.

    `classical_cv.defect_images.build_demo_dataset` orders its output as
    ALL positives (`"positive_%04d"`) followed by ALL negatives
    (`"negative_%04d"`) -- a plain `images[:-k]`/`images[-k:]` slice would
    put only negatives (zero ground truth boxes) in a tail-sized val
    split, making `average_precision` trivially `0.0` for every class
    regardless of how well the detector actually works (caught empirically
    while building this week's demo -- see `modern_detectors/README.md`'s
    Honesty note). Splitting positives and negatives separately and
    concatenating avoids that.

    Args:
        images: A full dataset (positives then negatives, or any order --
            this only groups by "has at least one ground truth box").
        val_fraction: Fraction of EACH group (positive, negative) held
            out for validation.

    Returns:
        `(train_images, val_images)`.
    """
    positives = [image for image in images if image.ground_truth_boxes]
    negatives = [image for image in images if not image.ground_truth_boxes]

    def _split(group: list[ResizedImage]) -> tuple[list[ResizedImage], list[ResizedImage]]:
        if not group:
            return [], []
        num_val = max(1, round(len(group) * val_fraction))
        return group[:-num_val], group[-num_val:]

    train_positives, val_positives = _split(positives)
    train_negatives, val_negatives = _split(negatives)
    return train_positives + train_negatives, val_positives + val_negatives


def write_yolo_dataset(images: list[ResizedImage], output_dir: str | Path) -> Path:
    """Write `images` as a YOLO-format directory tree + `data.yaml`.

    Args:
        images: Resized images (see `resize_dataset`) -- written as a
            single split, both `train:` and `val:` in the written
            `data.yaml` point at the same directory (week 26's own
            `benchmark.py` does its own held-out split for scoring,
            independently of whatever `ultralytics` reports internally).
        output_dir: Directory to write `images/`, `labels/`, `data.yaml` under.

    Returns:
        Path to the written `data.yaml`.
    """
    root = Path(output_dir).resolve()
    images_dir = root / "images"
    labels_dir = root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for image in images:
        Image.fromarray(image.rgb_pixels).save(images_dir / f"{image.image_id}.jpg")

        lines = []
        for x0, y0, x1, y1 in image.ground_truth_boxes:
            cx = (x0 + x1) / 2 / TRAIN_IMAGE_SIZE
            cy = (y0 + y1) / 2 / TRAIN_IMAGE_SIZE
            box_w = (x1 - x0) / TRAIN_IMAGE_SIZE
            box_h = (y1 - y0) / TRAIN_IMAGE_SIZE
            lines.append(f"{DEFECT_CLASS_INDEX} {cx:.6f} {cy:.6f} {box_w:.6f} {box_h:.6f}")
        (labels_dir / f"{image.image_id}.txt").write_text("\n".join(lines), encoding="utf-8")

    data_yaml_path = root / "data.yaml"
    data_yaml_path.write_text(
        f'train: {images_dir}\nval: {images_dir}\nnc: 1\nnames: ["{DEFECT_CLASS_NAME}"]\n',
        encoding="utf-8",
    )
    return data_yaml_path
