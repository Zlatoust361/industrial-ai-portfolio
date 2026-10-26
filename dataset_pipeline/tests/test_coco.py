"""Unit tests for `utils.coco`."""

from __future__ import annotations

import json
from pathlib import Path

from utils.coco import build_coco_dataset, to_coco_bbox, write_coco_dataset
from utils.manifest import Sample

OK_SAMPLE = Sample(
    sample_id="s0",
    image_path="images/s0.jpg",
    width=100,
    height=80,
    label="ok",
    object_bbox=(10, 10, 60, 50),
    defect_boxes=(),
    dr_params={},
)
DEFECT_SAMPLE = Sample(
    sample_id="s1",
    image_path="images/s1.jpg",
    width=100,
    height=80,
    label="defect",
    object_bbox=(5, 5, 40, 40),
    defect_boxes=((12, 12, 20, 20), (25, 25, 30, 30)),
    dr_params={},
)


def test_to_coco_bbox_converts_xyxy_to_xywh() -> None:
    assert to_coco_bbox((10, 10, 60, 50)) == [10.0, 10.0, 50.0, 40.0]


def test_build_coco_dataset_has_two_categories() -> None:
    dataset = build_coco_dataset([OK_SAMPLE], "part", "defect")

    assert dataset["categories"] == [{"id": 1, "name": "part"}, {"id": 2, "name": "defect"}]


def test_build_coco_dataset_one_image_and_object_annotation_for_ok_sample() -> None:
    dataset = build_coco_dataset([OK_SAMPLE], "part", "defect")

    assert len(dataset["images"]) == 1
    assert dataset["images"][0]["file_name"] == "images/s0.jpg"
    assert len(dataset["annotations"]) == 1
    assert dataset["annotations"][0]["category_id"] == 1


def test_build_coco_dataset_adds_defect_annotations() -> None:
    dataset = build_coco_dataset([DEFECT_SAMPLE], "part", "defect")

    # 1 object annotation + 2 defect annotations.
    assert len(dataset["annotations"]) == 3
    defect_annotations = [a for a in dataset["annotations"] if a["category_id"] == 2]
    assert len(defect_annotations) == 2


def test_build_coco_dataset_annotation_ids_are_unique_across_images() -> None:
    dataset = build_coco_dataset([OK_SAMPLE, DEFECT_SAMPLE], "part", "defect")

    ids = [a["id"] for a in dataset["annotations"]]
    assert len(ids) == len(set(ids))


def test_write_coco_dataset_creates_parent_dirs_and_valid_json(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "annotations.json"

    result = write_coco_dataset([OK_SAMPLE], "part", "defect", output_path)

    assert result == output_path
    payload = json.loads(output_path.read_text())
    assert payload["images"][0]["file_name"] == "images/s0.jpg"
