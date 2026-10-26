"""Unit tests for `utils.manifest`."""

from __future__ import annotations

from pathlib import Path

import pytest
from utils.manifest import (
    Sample,
    load_manifest,
    sample_from_dict,
    save_manifest,
    with_split,
    with_stratum,
)

SAMPLE = Sample(
    sample_id="sample_000000",
    image_path="images/sample_000000.jpg",
    width=512,
    height=512,
    label="defect",
    object_bbox=(10, 10, 100, 100),
    defect_boxes=((20, 20, 40, 40),),
    dr_params={"light": {"brightness": 1.0}},
)


def test_sample_round_trips_through_dict() -> None:
    restored = sample_from_dict(SAMPLE.to_dict())

    assert restored == SAMPLE


def test_save_and_load_manifest_round_trips(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"

    save_manifest(manifest_path, [SAMPLE])
    loaded = load_manifest(manifest_path)

    assert loaded == [SAMPLE]


def test_save_manifest_creates_parent_dirs(tmp_path: Path) -> None:
    manifest_path = tmp_path / "nested" / "dir" / "manifest.json"

    save_manifest(manifest_path, [SAMPLE])

    assert manifest_path.is_file()


def test_load_manifest_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        load_manifest(tmp_path / "missing.json")


def test_with_stratum_returns_new_immutable_copy() -> None:
    updated = with_stratum(SAMPLE, "defect_low_left_dark")

    assert updated.stratum == "defect_low_left_dark"
    assert SAMPLE.stratum is None  # original untouched


def test_with_split_returns_new_immutable_copy() -> None:
    updated = with_split(SAMPLE, "train")

    assert updated.split == "train"
    assert SAMPLE.split is None
