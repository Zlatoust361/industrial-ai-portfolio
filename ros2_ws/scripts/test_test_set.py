"""Unit tests for test_set.py.

Only needs PIL (to write tiny synthetic images) -- no torch/model
dependency. Run with:
    cd ros2_ws/scripts && pytest test_test_set.py
"""

from pathlib import Path

import pytest
from PIL import Image
from test_set import load_test_set, parse_yolo_label_line


def test_parse_yolo_label_line_converts_normalized_to_pixel_xyxy() -> None:
    # class 0, centered, half the image's width/height -> a centered quarter-area box.
    gt = parse_yolo_label_line(
        "0 0.5 0.5 0.5 0.5", class_names=["cat"], image_width=100, image_height=100, image_id="img0"
    )
    assert gt is not None
    assert gt.class_name == "cat"
    assert gt.image_id == "img0"
    assert gt.box == pytest.approx((25.0, 25.0, 75.0, 75.0))


def test_parse_yolo_label_line_skips_blank_and_comment_lines() -> None:
    assert parse_yolo_label_line("", ["cat"], 100, 100, "img0") is None
    assert parse_yolo_label_line("   ", ["cat"], 100, 100, "img0") is None
    assert parse_yolo_label_line("# a comment", ["cat"], 100, 100, "img0") is None


def test_parse_yolo_label_line_rejects_bad_field_count() -> None:
    with pytest.raises(ValueError, match="class_id cx cy w h"):
        parse_yolo_label_line("0 0.5 0.5", ["cat"], 100, 100, "img0")


def test_parse_yolo_label_line_rejects_out_of_range_class_id() -> None:
    with pytest.raises(ValueError, match="out of range"):
        parse_yolo_label_line("5 0.5 0.5 0.5 0.5", ["cat"], 100, 100, "img0")


def test_load_test_set_reads_images_labels_and_classes(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    (tmp_path / "classes.txt").write_text("cat\ndog\n")

    Image.new("RGB", (100, 100)).save(tmp_path / "images" / "a.jpg")
    (tmp_path / "labels" / "a.txt").write_text("0 0.5 0.5 0.5 0.5\n")

    image_paths, ground_truths, class_names = load_test_set(tmp_path)

    assert class_names == ["cat", "dog"]
    assert [p.name for p in image_paths] == ["a.jpg"]
    assert len(ground_truths) == 1
    assert ground_truths[0].class_name == "cat"
    assert ground_truths[0].image_id == "a.jpg"


def test_load_test_set_skips_images_without_matching_labels(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    (tmp_path / "classes.txt").write_text("cat\n")

    Image.new("RGB", (10, 10)).save(tmp_path / "images" / "unlabeled.jpg")

    image_paths, ground_truths, _ = load_test_set(tmp_path)

    assert len(image_paths) == 1  # still run detection on it
    assert len(ground_truths) == 0  # but no ground-truth boxes


def test_load_test_set_missing_images_dir_raises(tmp_path: Path) -> None:
    (tmp_path / "classes.txt").write_text("cat\n")
    with pytest.raises(FileNotFoundError, match="images"):
        load_test_set(tmp_path)


def test_load_test_set_missing_classes_file_raises(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir()
    with pytest.raises(FileNotFoundError, match="classes.txt"):
        load_test_set(tmp_path)
