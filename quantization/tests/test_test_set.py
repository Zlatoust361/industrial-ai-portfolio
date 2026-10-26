"""Unit tests for `utils/test_set.py`."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image
from utils.test_set import discover_test_images


def _make_images(directory: Path, count: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        Image.new("RGB", (4, 4)).save(directory / f"img_{index:03d}.jpg")


def test_discover_test_images_sorted_and_uncapped(tmp_path: Path) -> None:
    _make_images(tmp_path, count=5)

    images = discover_test_images(tmp_path)

    assert len(images) == 5
    assert images == sorted(images)


def test_discover_test_images_capped(tmp_path: Path) -> None:
    _make_images(tmp_path, count=10)

    images = discover_test_images(tmp_path, max_images=3)

    assert len(images) == 3
    assert images == sorted(tmp_path.glob("*.jpg"))[:3]


def test_discover_test_images_deterministic_across_calls(tmp_path: Path) -> None:
    _make_images(tmp_path, count=10)

    first = discover_test_images(tmp_path, max_images=4)
    second = discover_test_images(tmp_path, max_images=4)

    assert first == second


def test_discover_test_images_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="dataset_pipeline"):
        discover_test_images(tmp_path / "does_not_exist")


def test_discover_test_images_empty_dir_raises(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()

    with pytest.raises(ValueError, match="No test images"):
        discover_test_images(tmp_path / "empty")
