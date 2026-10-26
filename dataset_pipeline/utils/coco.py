"""Manifest -> COCO-format annotations (week 06, stage 4).

Two categories: the config's `object_class_name` (always one bbox per
sample, the rendered object) and `defect_class_name` (zero or more boxes,
only for `label == "defect"`). Day 3 adds a dataset-integrity validator
alongside this and `utils/yolo.py`; this module's own output is already a
real, spec-shaped COCO file -- `images`/`annotations`/`categories`, bbox
in COCO's `[x, y, width, height]` convention (not the `(x_min, y_min,
x_max, y_max)` this pipeline uses internally).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from utils.manifest import Bbox, Sample


def to_coco_bbox(box: Bbox) -> list[float]:
    """Convert an `(x_min, y_min, x_max, y_max)` box to COCO's `[x, y, w, h]`.

    Args:
        box: A pixel-space box.

    Returns:
        `[x_min, y_min, width, height]`, all floats.
    """
    x0, y0, x1, y1 = box
    return [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]


def build_coco_dataset(
    samples: list[Sample], object_class_name: str, defect_class_name: str
) -> dict[str, Any]:
    """Build a COCO-format dict from `samples`.

    Args:
        samples: Samples to include (pre-filtered by split, if applicable).
        object_class_name: Category name for each sample's `object_bbox`.
        defect_class_name: Category name for `defect_boxes` entries.

    Returns:
        `{"images": [...], "annotations": [...], "categories": [...]}`,
        category id `1` = `object_class_name`, `2` = `defect_class_name`.
    """
    categories = [
        {"id": 1, "name": object_class_name},
        {"id": 2, "name": defect_class_name},
    ]
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    next_annotation_id = 1

    for image_id, sample in enumerate(samples, start=1):
        images.append(
            {
                "id": image_id,
                "file_name": sample.image_path,
                "width": sample.width,
                "height": sample.height,
            }
        )

        object_bbox = to_coco_bbox(sample.object_bbox)
        annotations.append(
            {
                "id": next_annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": object_bbox,
                "area": object_bbox[2] * object_bbox[3],
                "iscrowd": 0,
            }
        )
        next_annotation_id += 1

        for defect_box in sample.defect_boxes:
            bbox = to_coco_bbox(defect_box)
            annotations.append(
                {
                    "id": next_annotation_id,
                    "image_id": image_id,
                    "category_id": 2,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                }
            )
            next_annotation_id += 1

    return {"images": images, "annotations": annotations, "categories": categories}


def write_coco_dataset(
    samples: list[Sample],
    object_class_name: str,
    defect_class_name: str,
    output_path: str | Path,
) -> Path:
    """Build and write a COCO JSON file, creating parent directories.

    Args:
        samples: Samples to include.
        object_class_name: Category name for `object_bbox`.
        defect_class_name: Category name for `defect_boxes`.
        output_path: Destination JSON file path.

    Returns:
        `output_path`, as a `Path`.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = build_coco_dataset(samples, object_class_name, defect_class_name)
    output_path.write_text(json.dumps(dataset, indent=2))
    return output_path
