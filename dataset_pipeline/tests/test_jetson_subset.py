"""Unit tests for `utils.jetson_subset`."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image
from utils.config import JetsonSubsetConfig
from utils.jetson_subset import export_jetson_subset, select_stratified_subset
from utils.manifest import Sample, with_split

CONFIG = JetsonSubsetConfig(min_count=20, target_count=30, max_count=40, seed=0)


def _make_val_samples(n: int, stratum: str) -> list[Sample]:
    return [
        with_split(
            Sample(
                sample_id=f"{stratum}_{i}",
                image_path=f"images/{stratum}_{i}.jpg",
                width=32,
                height=32,
                label="ok",
                object_bbox=(0, 0, 10, 10),
                defect_boxes=(),
                dr_params={},
                stratum=stratum,
            ),
            "val",
        )
        for i in range(n)
    ]


def test_select_stratified_subset_size_within_bounds() -> None:
    samples = _make_val_samples(60, "a") + _make_val_samples(40, "b")

    subset = select_stratified_subset(samples, CONFIG)

    assert CONFIG.min_count <= len(subset) <= CONFIG.max_count
    assert len(subset) == CONFIG.target_count


def test_select_stratified_subset_keeps_every_stratum_represented() -> None:
    rare = _make_val_samples(1, "rare")[0]
    samples = _make_val_samples(60, "a") + _make_val_samples(40, "b") + [rare]

    subset = select_stratified_subset(samples, CONFIG)

    strata_present = {sample.stratum for sample in subset}
    assert {"a", "b"} <= strata_present


def test_select_stratified_subset_no_duplicates() -> None:
    samples = _make_val_samples(60, "a") + _make_val_samples(40, "b")

    subset = select_stratified_subset(samples, CONFIG)

    ids = [sample.sample_id for sample in subset]
    assert len(ids) == len(set(ids))


def test_select_stratified_subset_empty_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        select_stratified_subset([], CONFIG)


def test_select_stratified_subset_too_few_samples_raises() -> None:
    with pytest.raises(ValueError, match="need at least"):
        select_stratified_subset(_make_val_samples(5, "a"), CONFIG)


def _write_val_yolo_dir(root: Path, sample_ids: list[str]) -> None:
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    (root / "classes.txt").write_text("part\ndefect\n")
    for sample_id in sample_ids:
        Image.new("RGB", (8, 8)).save(root / "images" / f"{sample_id}.jpg")
        (root / "labels" / f"{sample_id}.txt").write_text("0 0.5 0.5 0.2 0.2\n")


def test_export_jetson_subset_copies_files_and_reports_size(tmp_path: Path) -> None:
    val_yolo_dir = tmp_path / "val_yolo"
    _write_val_yolo_dir(val_yolo_dir, ["a_0", "a_1"])
    subset = _make_val_samples(2, "a")

    output_dir, total_bytes = export_jetson_subset(subset, val_yolo_dir, tmp_path / "jetson_subset")

    assert (output_dir / "images" / "a_0.jpg").is_file()
    assert (output_dir / "labels" / "a_1.txt").is_file()
    assert (output_dir / "classes.txt").is_file()
    assert total_bytes > 0


def test_export_jetson_subset_missing_file_raises(tmp_path: Path) -> None:
    val_yolo_dir = tmp_path / "val_yolo"
    _write_val_yolo_dir(val_yolo_dir, [])
    (val_yolo_dir / "classes.txt").write_text("part\ndefect\n")
    subset = _make_val_samples(1, "a")

    with pytest.raises(FileNotFoundError, match="Missing YOLO"):
        export_jetson_subset(subset, val_yolo_dir, tmp_path / "jetson_subset")
