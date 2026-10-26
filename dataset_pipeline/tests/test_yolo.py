"""Unit tests for `utils.yolo` -- real files on disk, no mocking."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image
from utils.manifest import Sample
from utils.yolo import to_yolo_line, write_yolo_dataset


def test_to_yolo_line_normalizes_and_centers() -> None:
    line = to_yolo_line(0, (10, 10, 30, 50), width=100, height=100)

    class_id, cx, cy, w, h = line.split()
    assert class_id == "0"
    assert float(cx) == pytest.approx(0.20)
    assert float(cy) == pytest.approx(0.30)
    assert float(w) == pytest.approx(0.20)
    assert float(h) == pytest.approx(0.40)


def _write_source_image(path: Path, size: tuple[int, int] = (64, 48)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(10, 20, 30)).save(path)


def test_write_yolo_dataset_writes_images_labels_and_classes(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    _write_source_image(source_dir / "images" / "s0.jpg", size=(64, 48))

    sample = Sample(
        sample_id="s0",
        image_path="images/s0.jpg",
        width=64,
        height=48,
        label="defect",
        object_bbox=(0, 0, 32, 24),
        defect_boxes=((10, 10, 20, 20),),
        dr_params={},
    )

    output_dir = write_yolo_dataset([sample], source_dir, "part", "defect", tmp_path / "yolo_out")

    assert (output_dir / "classes.txt").read_text() == "part\ndefect\n"
    assert (output_dir / "images" / "s0.jpg").is_file()
    label_lines = (output_dir / "labels" / "s0.txt").read_text().strip().splitlines()
    assert len(label_lines) == 2  # 1 object + 1 defect box
    assert label_lines[0].startswith("0 ")
    assert label_lines[1].startswith("1 ")


def test_write_yolo_dataset_ok_sample_has_only_object_line(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    _write_source_image(source_dir / "images" / "s1.jpg")

    sample = Sample(
        sample_id="s1",
        image_path="images/s1.jpg",
        width=64,
        height=48,
        label="ok",
        object_bbox=(0, 0, 32, 24),
        defect_boxes=(),
        dr_params={},
    )

    output_dir = write_yolo_dataset([sample], source_dir, "part", "defect", tmp_path / "yolo_out")

    label_lines = (output_dir / "labels" / "s1.txt").read_text().strip().splitlines()
    assert len(label_lines) == 1
    assert label_lines[0].startswith("0 ")


def test_write_yolo_dataset_missing_source_image_raises(tmp_path: Path) -> None:
    sample = Sample(
        sample_id="s2",
        image_path="images/missing.jpg",
        width=64,
        height=48,
        label="ok",
        object_bbox=(0, 0, 32, 24),
        defect_boxes=(),
        dr_params={},
    )

    with pytest.raises(FileNotFoundError, match="not found"):
        write_yolo_dataset([sample], tmp_path / "source", "part", "defect", tmp_path / "yolo_out")
