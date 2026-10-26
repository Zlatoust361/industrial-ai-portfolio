"""Binary "has a defect" classification dataset for week 28, built on week 25's real images.

Reuses `classical_cv.defect_images.build_demo_dataset` as-is (week 25's
real generator, itself built on week 05's real `generate_defect_mask`)
-- week 28 does not invent a new defect dataset. What's new here is
purely the task framing: week 25/26 localize defects (bounding boxes),
week 28 only needs a much cheaper binary label ("does this image have
a defect at all") -- model compression technique comparisons don't need
a harder task than that to be real and informative.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from classical_cv.defect_images import GeneratedImage, build_demo_dataset
from src.synthetic.config import DefectsConfig

CLASSIFICATION_IMAGE_SIZE = 64
"""Square resolution every image is downsized to -- small enough that a
tiny CNN's forward pass takes a fraction of a millisecond on CPU, the
whole point of this week's models being toy-scale and fast to train
dozens of times over (NAS search, distillation, pruning, all in one run)."""


@dataclass(frozen=True)
class ClassificationDataset:
    """A ready-to-train tensor dataset.

    Attributes:
        images: `(N, 1, H, W)` float32 tensor, pixel values in `[0, 1]`.
        labels: `(N, 1)` float32 tensor, `1.0` if the source image has at
            least one real ground truth defect, else `0.0`.
        image_ids: Source `GeneratedImage.image_id`, same order.
    """

    images: torch.Tensor
    labels: torch.Tensor
    image_ids: tuple[str, ...]


def _resize_to_tensor(image: GeneratedImage, size: int) -> torch.Tensor:
    """Grayscale `GeneratedImage.pixels` -> `(1, size, size)` float32 tensor in `[0, 1]`."""
    resized = Image.fromarray(image.pixels, mode="L").resize((size, size), resample=Image.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def build_classification_dataset(
    images: list[GeneratedImage], size: int = CLASSIFICATION_IMAGE_SIZE
) -> ClassificationDataset:
    """Convert a list of week 25 `GeneratedImage`s into a tensor classification dataset.

    Args:
        images: Images from `classical_cv.defect_images.build_demo_dataset`.
        size: Target square resolution.

    Returns:
        A `ClassificationDataset` with one label per image.
    """
    tensors = torch.stack([_resize_to_tensor(image, size) for image in images])
    labels = torch.tensor(
        [[1.0 if image.ground_truth_boxes else 0.0] for image in images], dtype=torch.float32
    )
    return ClassificationDataset(
        images=tensors, labels=labels, image_ids=tuple(image.image_id for image in images)
    )


def build_train_val_split(
    config: DefectsConfig,
    num_positive: int,
    num_negative: int,
    val_fraction: float,
    seed: int,
    size: int = CLASSIFICATION_IMAGE_SIZE,
) -> tuple[ClassificationDataset, ClassificationDataset]:
    """Build a stratified train/val split of the binary defect-classification task.

    Args:
        config: Parsed `config/defects.yaml` (week 05's real shapes).
        num_positive: Images with 1-2 real defects each.
        num_negative: Clean images.
        val_fraction: Fraction of EACH group (positive, negative) held out
            for validation -- same stratification `modern_detectors.dataset
            .split_train_val` uses, for the same reason: `build_demo_dataset`
            orders all positives before all negatives, so a naive tail
            slice would put only one class in val.
        seed: RNG seed -- reproducible dataset.
        size: Target square resolution.

    Returns:
        `(train_dataset, val_dataset)`.
    """
    raw = build_demo_dataset(config, num_positive, num_negative, seed)
    positives = [image for image in raw if image.ground_truth_boxes]
    negatives = [image for image in raw if not image.ground_truth_boxes]

    def _split(group: list[GeneratedImage]) -> tuple[list[GeneratedImage], list[GeneratedImage]]:
        if not group:
            return [], []
        num_val = max(1, round(len(group) * val_fraction))
        return group[:-num_val], group[-num_val:]

    train_positives, val_positives = _split(positives)
    train_negatives, val_negatives = _split(negatives)

    train_images = build_classification_dataset(train_positives + train_negatives, size)
    val_images = build_classification_dataset(val_positives + val_negatives, size)
    return train_images, val_images
