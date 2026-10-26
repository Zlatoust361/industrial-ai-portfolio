"""Unit tests for `utils.split`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image
from utils.config import SplitConfig
from utils.manifest import Sample, with_split
from utils.split import (
    SPLITS,
    assign_splits,
    partition_coco_by_split,
    reorganize_yolo_by_split,
    split_distribution,
)

CONFIG = SplitConfig(train_fraction=0.7, val_fraction=0.15, test_fraction=0.15, seed=0)


def _make_sample(sample_id: str, stratum: str, source_sample_id: str | None = None) -> Sample:
    return Sample(
        sample_id=sample_id,
        image_path=f"images/{sample_id}.jpg",
        width=32,
        height=32,
        label="ok",
        object_bbox=(0, 0, 10, 10),
        defect_boxes=(),
        dr_params={},
        stratum=stratum,
        source_sample_id=source_sample_id,
    )


def test_assign_splits_covers_every_sample_with_a_valid_split() -> None:
    samples = [_make_sample(f"s{i}", "ok_mid_center_light") for i in range(20)]

    result = assign_splits(samples, CONFIG)

    assert len(result) == 20
    assert all(sample.split in SPLITS for sample in result)


def test_assign_splits_matches_configured_fractions_for_a_large_single_stratum() -> None:
    samples = [_make_sample(f"s{i}", "ok_mid_center_light") for i in range(100)]

    result = assign_splits(samples, CONFIG)

    counts = {split: sum(1 for s in result if s.split == split) for split in SPLITS}
    assert counts["train"] == 70
    assert counts["val"] == 15
    assert counts["test"] == 15


def test_assign_splits_single_sample_stratum_goes_to_train_not_always_test() -> None:
    # A naive "dump the leftover into test" implementation would always
    # send a 1-sample stratum to test; largest-remainder correctly sends
    # it to whichever split has the largest configured fraction (train).
    samples = [_make_sample("solo", "rare_stratum")]

    result = assign_splits(samples, CONFIG)

    assert result[0].split == "train"


def test_assign_splits_augmented_sample_inherits_source_split() -> None:
    base = [_make_sample(f"s{i}", "ok_mid_center_light") for i in range(20)]
    augmented = _make_sample("s0_aug", "ok_mid_center_light", source_sample_id="s0")

    result = assign_splits([*base, augmented], CONFIG)

    result_by_id = {s.sample_id: s for s in result}
    assert result_by_id["s0_aug"].split == result_by_id["s0"].split


def test_assign_splits_unknown_source_sample_id_raises() -> None:
    orphan = _make_sample("aug", "ok_mid_center_light", source_sample_id="does_not_exist")

    with pytest.raises(ValueError, match="unknown source_sample_id"):
        assign_splits([orphan], CONFIG)


def test_split_distribution_counts_per_split_and_stratum() -> None:
    samples = assign_splits(
        [_make_sample(f"s{i}", "ok_mid_center_light") for i in range(10)], CONFIG
    )

    distribution = split_distribution(samples)

    assert set(distribution) == set(SPLITS)
    assert sum(distribution["train"].values()) + sum(distribution["val"].values()) + sum(
        distribution["test"].values()
    ) == 10


def _write_yolo_flat_dataset(root: Path, sample_ids: list[str]) -> None:
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    (root / "classes.txt").write_text("part\ndefect\n")
    for sample_id in sample_ids:
        Image.new("RGB", (8, 8)).save(root / "images" / f"{sample_id}.jpg")
        (root / "labels" / f"{sample_id}.txt").write_text("0 0.5 0.5 0.2 0.2\n")


def test_reorganize_yolo_by_split_moves_files_into_split_subdirs(tmp_path: Path) -> None:
    yolo_dir = tmp_path / "yolo"
    _write_yolo_flat_dataset(yolo_dir, ["s0", "s1"])
    samples = [
        with_split(_make_sample("s0", "a"), "train"),
        with_split(_make_sample("s1", "a"), "val"),
    ]

    reorganize_yolo_by_split(yolo_dir, samples)

    assert (yolo_dir / "train" / "images" / "s0.jpg").is_file()
    assert (yolo_dir / "train" / "labels" / "s0.txt").is_file()
    assert (yolo_dir / "train" / "classes.txt").read_text() == "part\ndefect\n"
    assert (yolo_dir / "val" / "images" / "s1.jpg").is_file()
    assert not (yolo_dir / "images").exists()
    assert not (yolo_dir / "labels").exists()


def test_reorganize_yolo_by_split_missing_file_raises(tmp_path: Path) -> None:
    yolo_dir = tmp_path / "yolo"
    _write_yolo_flat_dataset(yolo_dir, ["s0"])
    sample = with_split(_make_sample("missing", "a"), "train")

    with pytest.raises(FileNotFoundError, match="Missing YOLO"):
        reorganize_yolo_by_split(yolo_dir, [sample])


def test_partition_coco_by_split_filters_by_filename(tmp_path: Path) -> None:
    coco_path = tmp_path / "annotations.json"
    coco_path.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "images/s0.jpg", "width": 8, "height": 8},
                    {"id": 2, "file_name": "images/s1.jpg", "width": 8, "height": 8},
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [0, 0, 4, 4],
                        "area": 16,
                        "iscrowd": 0,
                    },
                    {
                        "id": 2,
                        "image_id": 2,
                        "category_id": 1,
                        "bbox": [0, 0, 4, 4],
                        "area": 16,
                        "iscrowd": 0,
                    },
                ],
                "categories": [{"id": 1, "name": "part"}],
            }
        )
    )
    s0 = with_split(_make_sample("s0", "a"), "train")
    s1 = with_split(_make_sample("s1", "a"), "val")

    output_dir = tmp_path / "split_coco"
    partition_coco_by_split(coco_path, [s0, s1], output_dir)

    train_data = json.loads((output_dir / "annotations_train.json").read_text())
    val_data = json.loads((output_dir / "annotations_val.json").read_text())
    test_data = json.loads((output_dir / "annotations_test.json").read_text())
    assert [img["file_name"] for img in train_data["images"]] == ["images/s0.jpg"]
    assert [img["file_name"] for img in val_data["images"]] == ["images/s1.jpg"]
    assert test_data["images"] == []


def test_partition_coco_by_split_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        partition_coco_by_split(tmp_path / "missing.json", [], tmp_path / "out")
