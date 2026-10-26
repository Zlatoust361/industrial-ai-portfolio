"""Unit tests for `distillation.dataset`."""

from __future__ import annotations

from distillation.dataset import build_classification_dataset, build_train_val_split
from src.synthetic.config import load_defects_config

DEFECTS_CONFIG = load_defects_config("config/defects.yaml")


def test_build_classification_dataset_shapes_and_labels() -> None:
    from classical_cv.defect_images import build_demo_dataset

    images = build_demo_dataset(DEFECTS_CONFIG, num_positive=3, num_negative=2, seed=42)
    dataset = build_classification_dataset(images, size=32)

    assert dataset.images.shape == (5, 1, 32, 32)
    assert dataset.labels.shape == (5, 1)
    assert dataset.images.min() >= 0.0
    assert dataset.images.max() <= 1.0
    assert dataset.labels[:3].sum() == 3  # positives first, build_demo_dataset's own order
    assert dataset.labels[3:].sum() == 0


def test_build_train_val_split_is_stratified() -> None:
    """Same bug class `modern_detectors.dataset.split_train_val` guards
    against (week 26): `build_demo_dataset` orders all positives before
    all negatives, so val must get both classes, not just the tail class."""
    train, val = build_train_val_split(
        DEFECTS_CONFIG, num_positive=20, num_negative=10, val_fraction=0.25, seed=42, size=32
    )
    assert val.labels.sum() > 0
    assert (val.labels == 0).sum() > 0
    assert len(train.images) + len(val.images) == 30


def test_build_train_val_split_is_deterministic_for_the_same_seed() -> None:
    import torch

    train_a, val_a = build_train_val_split(
        DEFECTS_CONFIG, num_positive=5, num_negative=5, val_fraction=0.2, seed=7, size=32
    )
    train_b, val_b = build_train_val_split(
        DEFECTS_CONFIG, num_positive=5, num_negative=5, val_fraction=0.2, seed=7, size=32
    )
    torch.testing.assert_close(train_a.images, train_b.images)
    torch.testing.assert_close(val_a.images, val_b.images)
