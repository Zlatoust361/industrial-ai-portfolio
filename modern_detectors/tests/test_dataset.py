"""Unit tests for `modern_detectors.dataset`."""

from __future__ import annotations

import numpy as np

from classical_cv.defect_images import build_demo_dataset
from modern_detectors.dataset import (
    TRAIN_IMAGE_SIZE,
    resize_dataset,
    resize_image,
    split_train_val,
    write_yolo_dataset,
)
from src.synthetic.config import load_defects_config

DEFECTS_CONFIG = load_defects_config("config/defects.yaml")
SOURCE_SIZE = (512, 512)


def test_resize_image_shape_and_dtype() -> None:
    dataset = build_demo_dataset(DEFECTS_CONFIG, num_positive=1, num_negative=0, seed=42)
    resized = resize_image(dataset[0], SOURCE_SIZE)
    assert resized.rgb_pixels.shape == (TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE, 3)
    assert resized.rgb_pixels.dtype == np.uint8


def test_resize_image_rescales_boxes_proportionally() -> None:
    dataset = build_demo_dataset(DEFECTS_CONFIG, num_positive=1, num_negative=0, seed=42)
    source = dataset[0]
    resized = resize_image(source, SOURCE_SIZE)

    scale = TRAIN_IMAGE_SIZE / SOURCE_SIZE[0]
    box_pairs = zip(source.ground_truth_boxes, resized.ground_truth_boxes, strict=True)
    for source_box, resized_box in box_pairs:
        for source_coord, resized_coord in zip(source_box, resized_box, strict=True):
            assert resized_coord == source_coord * scale


def test_resize_dataset_preserves_order_and_count() -> None:
    dataset = build_demo_dataset(DEFECTS_CONFIG, num_positive=3, num_negative=2, seed=42)
    resized = resize_dataset(dataset, SOURCE_SIZE)
    assert [image.image_id for image in resized] == [image.image_id for image in dataset]


def test_split_train_val_keeps_both_groups_in_val() -> None:
    """The bug this guards against: `build_demo_dataset` orders ALL positives
    before ALL negatives, so a naive tail slice puts only negatives (zero
    ground truth) in val -- see `modern_detectors/README.md`'s Honesty note."""
    dataset = build_demo_dataset(DEFECTS_CONFIG, num_positive=20, num_negative=10, seed=42)
    resized = resize_dataset(dataset, SOURCE_SIZE)

    train_images, val_images = split_train_val(resized, val_fraction=0.25)

    val_positives = [image for image in val_images if image.ground_truth_boxes]
    val_negatives = [image for image in val_images if not image.ground_truth_boxes]
    assert val_positives, "val split must include at least one positive image"
    assert val_negatives, "val split must include at least one negative image"
    assert len(train_images) + len(val_images) == len(resized)


def test_split_train_val_is_disjoint() -> None:
    dataset = build_demo_dataset(DEFECTS_CONFIG, num_positive=10, num_negative=10, seed=1)
    resized = resize_dataset(dataset, SOURCE_SIZE)
    train_images, val_images = split_train_val(resized, val_fraction=0.3)

    train_ids = {image.image_id for image in train_images}
    val_ids = {image.image_id for image in val_images}
    assert train_ids.isdisjoint(val_ids)


def test_write_yolo_dataset_writes_images_and_labels(tmp_path) -> None:
    dataset = build_demo_dataset(DEFECTS_CONFIG, num_positive=2, num_negative=1, seed=42)
    resized = resize_dataset(dataset, SOURCE_SIZE)

    data_yaml_path = write_yolo_dataset(resized, tmp_path / "yolo_ds")

    assert data_yaml_path.is_file()
    assert data_yaml_path.is_absolute()
    content = data_yaml_path.read_text(encoding="utf-8")
    assert "nc: 1" in content
    assert '"defect"' in content

    for image in resized:
        assert (tmp_path / "yolo_ds" / "images" / f"{image.image_id}.jpg").is_file()
        assert (tmp_path / "yolo_ds" / "labels" / f"{image.image_id}.txt").is_file()


def test_write_yolo_dataset_label_line_count_matches_boxes(tmp_path) -> None:
    dataset = build_demo_dataset(DEFECTS_CONFIG, num_positive=3, num_negative=1, seed=42)
    resized = resize_dataset(dataset, SOURCE_SIZE)
    write_yolo_dataset(resized, tmp_path / "yolo_ds")

    for image in resized:
        label_path = tmp_path / "yolo_ds" / "labels" / f"{image.image_id}.txt"
        lines = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line]
        assert len(lines) == len(image.ground_truth_boxes)
        for line in lines:
            parts = line.split()
            assert len(parts) == 5
            assert parts[0] == "0"
            for value in parts[1:]:
                assert 0.0 <= float(value) <= 1.0
