"""Unit tests for `src.data.dataset.ImageFolderDataset`/`LabeledImageFolderDataset`."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from src.data.dataset import ImageFolderDataset, LabeledImageFolderDataset, iter_batches


def test_image_folder_dataset_finds_only_image_files(tmp_path: Path) -> None:
    Image.new("RGB", (8, 8)).save(tmp_path / "a.png")
    Image.new("RGB", (8, 8)).save(tmp_path / "b.jpg")
    (tmp_path / "notes.txt").write_text("not an image")

    dataset = ImageFolderDataset(tmp_path)

    assert len(dataset) == 2
    assert dataset[0].load().size == (8, 8)


def test_labeled_image_folder_dataset_reads_class_subfolders(tmp_path: Path) -> None:
    (tmp_path / "ok").mkdir()
    (tmp_path / "defect").mkdir()
    Image.new("RGB", (8, 8)).save(tmp_path / "ok" / "a.png")
    Image.new("RGB", (8, 8)).save(tmp_path / "ok" / "b.png")
    Image.new("RGB", (8, 8)).save(tmp_path / "defect" / "c.png")

    dataset = LabeledImageFolderDataset(tmp_path, class_labels=["ok", "defect"])

    assert len(dataset) == 3
    labels = [sample.label for sample in dataset.samples]
    assert labels.count("ok") == 2
    assert labels.count("defect") == 1


def test_labeled_image_folder_dataset_missing_subfolder_raises(tmp_path: Path) -> None:
    (tmp_path / "ok").mkdir()

    with pytest.raises(FileNotFoundError, match="defect"):
        LabeledImageFolderDataset(tmp_path, class_labels=["ok", "defect"])


def test_iter_batches_splits_evenly() -> None:
    batches = list(iter_batches([1, 2, 3, 4], batch_size=2))
    assert batches == [[1, 2], [3, 4]]


def test_iter_batches_final_batch_may_be_shorter() -> None:
    batches = list(iter_batches([1, 2, 3, 4, 5], batch_size=2))
    assert batches == [[1, 2], [3, 4], [5]]


def test_iter_batches_batch_size_larger_than_items() -> None:
    batches = list(iter_batches([1, 2], batch_size=10))
    assert batches == [[1, 2]]


def test_iter_batches_empty_items() -> None:
    assert list(iter_batches([], batch_size=2)) == []
