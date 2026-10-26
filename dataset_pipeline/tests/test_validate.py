"""Unit tests for `utils.validate`."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from utils.manifest import Sample
from utils.validate import (
    find_bbox_issues,
    find_duplicate_issues,
    find_empty_mask_issues,
    validate_bbox,
    validate_dataset,
    validate_yolo_directory,
    validate_yolo_files,
)

VALID_SAMPLE = Sample(
    sample_id="s0",
    image_path="images/s0.jpg",
    width=100,
    height=100,
    label="defect",
    object_bbox=(0, 0, 50, 50),
    defect_boxes=((10, 10, 20, 20),),
    dr_params={},
)


def test_validate_bbox_accepts_a_valid_box() -> None:
    assert validate_bbox((0, 0, 50, 50), width=100, height=100) is None


def test_validate_bbox_rejects_degenerate_box() -> None:
    assert "degenerate" in validate_bbox((10, 10, 10, 20), width=100, height=100)


def test_validate_bbox_rejects_inverted_box() -> None:
    assert "degenerate" in validate_bbox((30, 10, 10, 20), width=100, height=100)


def test_validate_bbox_rejects_out_of_bounds_box() -> None:
    assert "out of bounds" in validate_bbox((0, 0, 150, 50), width=100, height=100)


def test_validate_bbox_rejects_negative_origin() -> None:
    assert "out of bounds" in validate_bbox((-5, 0, 50, 50), width=100, height=100)


def test_find_bbox_issues_flags_bad_object_bbox() -> None:
    bad_object = Sample(
        sample_id="bad_obj",
        image_path="images/bad_obj.jpg",
        width=100,
        height=100,
        label="ok",
        object_bbox=(10, 10, 10, 20),
        defect_boxes=(),
        dr_params={},
    )

    issues = find_bbox_issues([bad_object])

    assert len(issues) == 1
    assert issues[0].category == "bad_bbox"
    assert issues[0].sample_id == "bad_obj"


def test_find_bbox_issues_flags_bad_defect_box() -> None:
    sample = _make_sample("bad_defect", label="defect", defect_boxes=((200, 200, 210, 210),))

    issues = find_bbox_issues([sample])

    assert len(issues) == 1
    assert "defect_boxes" in issues[0].message


def test_find_bbox_issues_no_issues_for_valid_sample() -> None:
    assert find_bbox_issues([VALID_SAMPLE]) == []


def test_find_empty_mask_issues_flags_defect_with_no_boxes() -> None:
    sample = _make_sample("empty_defect", label="defect", defect_boxes=())

    issues = find_empty_mask_issues([sample])

    assert len(issues) == 1
    assert issues[0].category == "empty_defect_mask"


def test_find_empty_mask_issues_flags_ok_with_boxes() -> None:
    sample = _make_sample("bad_ok", label="ok", defect_boxes=((10, 10, 20, 20),))

    issues = find_empty_mask_issues([sample])

    assert len(issues) == 1
    assert issues[0].category == "unexpected_defect_boxes"


def test_find_empty_mask_issues_no_issues_for_consistent_samples() -> None:
    ok_sample = _make_sample("ok0", label="ok", defect_boxes=())
    defect_sample = _make_sample("defect0", label="defect", defect_boxes=((10, 10, 20, 20),))

    assert find_empty_mask_issues([ok_sample, defect_sample]) == []


def test_find_duplicate_issues_flags_duplicate_sample_id() -> None:
    a = _make_sample("dup", label="ok", defect_boxes=(), image_path="images/a.jpg")
    b = _make_sample("dup", label="ok", defect_boxes=(), image_path="images/b.jpg")

    issues = find_duplicate_issues([a, b])

    assert any(issue.category == "duplicate_sample_id" for issue in issues)


def test_find_duplicate_issues_flags_duplicate_image_path() -> None:
    a = _make_sample("s0", label="ok", defect_boxes=(), image_path="images/shared.jpg")
    b = _make_sample("s1", label="ok", defect_boxes=(), image_path="images/shared.jpg")

    issues = find_duplicate_issues([a, b])

    assert any(issue.category == "duplicate_image_path" for issue in issues)


def test_find_duplicate_issues_flags_duplicate_bbox_as_warning() -> None:
    sample = _make_sample(
        "dup_box", label="defect", defect_boxes=((10, 10, 20, 20), (10, 10, 20, 20))
    )

    issues = find_duplicate_issues([sample])

    assert len(issues) == 1
    assert issues[0].severity == "warning"
    assert issues[0].category == "duplicate_bbox"


def test_find_duplicate_issues_no_issues_for_clean_samples() -> None:
    a = _make_sample("s0", label="ok", defect_boxes=(), image_path="images/a.jpg")
    b = _make_sample("s1", label="ok", defect_boxes=(), image_path="images/b.jpg")

    assert find_duplicate_issues([a, b]) == []


def test_validate_dataset_clean_dataset_is_valid() -> None:
    report = validate_dataset([VALID_SAMPLE])

    assert report.is_valid
    assert report.errors == ()


def test_validate_dataset_aggregates_all_checks() -> None:
    bad_sample = _make_sample("bad", label="defect", defect_boxes=())  # empty mask

    report = validate_dataset([VALID_SAMPLE, bad_sample])

    assert not report.is_valid
    assert len(report.errors) == 1
    assert report.to_dict()["num_errors"] == 1


def _make_sample(
    sample_id: str, *, label: str, defect_boxes: tuple, image_path: str | None = None
) -> Sample:
    return Sample(
        sample_id=sample_id,
        image_path=image_path or f"images/{sample_id}.jpg",
        width=100,
        height=100,
        label=label,
        object_bbox=(0, 0, 50, 50),
        defect_boxes=defect_boxes,
        dr_params={},
    )


def _write_yolo_dir(root: Path, entries: dict[str, str]) -> None:
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    for stem, label_content in entries.items():
        Image.new("RGB", (8, 8)).save(root / "images" / f"{stem}.jpg")
        (root / "labels" / f"{stem}.txt").write_text(label_content)


def test_validate_yolo_files_no_issues_for_matching_dataset(tmp_path: Path) -> None:
    _write_yolo_dir(tmp_path, {"s0": "0 0.5 0.5 0.2 0.2\n"})
    sample = _make_sample("s0", label="ok", defect_boxes=())

    assert validate_yolo_files(tmp_path, [sample]) == []


def test_validate_yolo_files_flags_missing_image(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir(parents=True)
    (tmp_path / "labels").mkdir(parents=True)
    (tmp_path / "labels" / "s0.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    sample = _make_sample("s0", label="ok", defect_boxes=())

    issues = validate_yolo_files(tmp_path, [sample])

    assert any(
        issue.category == "missing_yolo_file" and "image" in issue.message for issue in issues
    )


def test_validate_yolo_files_flags_missing_label(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir(parents=True)
    (tmp_path / "labels").mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(tmp_path / "images" / "s0.jpg")
    sample = _make_sample("s0", label="ok", defect_boxes=())

    issues = validate_yolo_files(tmp_path, [sample])

    assert any(
        issue.category == "missing_yolo_file" and "label" in issue.message for issue in issues
    )


def test_validate_yolo_files_flags_out_of_range_coordinates(tmp_path: Path) -> None:
    _write_yolo_dir(tmp_path, {"s0": "0 1.5 0.5 0.2 0.2\n"})
    sample = _make_sample("s0", label="ok", defect_boxes=())

    issues = validate_yolo_files(tmp_path, [sample])

    assert any(issue.category == "bad_yolo_label_line" for issue in issues)


def test_validate_yolo_files_flags_wrong_field_count(tmp_path: Path) -> None:
    _write_yolo_dir(tmp_path, {"s0": "0 0.5 0.5 0.2\n"})
    sample = _make_sample("s0", label="ok", defect_boxes=())

    issues = validate_yolo_files(tmp_path, [sample])

    assert any(issue.category == "bad_yolo_label_line" for issue in issues)


def test_validate_yolo_directory_no_manifest_needed_and_clean_is_valid(tmp_path: Path) -> None:
    _write_yolo_dir(tmp_path, {"a": "0 0.5 0.5 0.2 0.2\n", "b": "1 0.3 0.3 0.1 0.1\n"})

    assert validate_yolo_directory(tmp_path) == []


def test_validate_yolo_directory_flags_image_with_no_label(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir(parents=True)
    (tmp_path / "labels").mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(tmp_path / "images" / "orphan.jpg")

    issues = validate_yolo_directory(tmp_path)

    assert len(issues) == 1
    assert issues[0].category == "missing_yolo_file"
    assert issues[0].sample_id == "orphan"


def test_validate_yolo_directory_flags_label_with_no_image(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir(parents=True)
    (tmp_path / "labels").mkdir(parents=True)
    (tmp_path / "labels" / "orphan.txt").write_text("0 0.5 0.5 0.2 0.2\n")

    issues = validate_yolo_directory(tmp_path)

    assert len(issues) == 1
    assert issues[0].category == "missing_yolo_file"


def test_validate_yolo_directory_flags_bad_label_line(tmp_path: Path) -> None:
    _write_yolo_dir(tmp_path, {"a": "0 1.5 0.5 0.2 0.2\n"})

    issues = validate_yolo_directory(tmp_path)

    assert any(issue.category == "bad_yolo_label_line" for issue in issues)
