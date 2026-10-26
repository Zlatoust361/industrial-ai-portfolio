"""Manifest -> YOLO-format dataset (week 06, stage 5): `images/`, `labels/*.txt`, `classes.txt`.

Same `class_id cx cy w h` normalized-line convention as
`src.evaluation.detection_metrics.parse_yolo_label_line`/`load_yolo_test_set`
(week 07) -- this pipeline's YOLO output is directly consumable by week
07's `train_dr_yolo.py`/`compare_dr_branches.py` (the "week 06 val subset"
those scripts' docstrings named but that didn't exist yet -- see
`course/DECISIONS.md` ADR-022), not a second, incompatible format.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from utils.manifest import Bbox, Sample


def to_yolo_line(class_id: int, box: Bbox, width: int, height: int) -> str:
    """Convert one pixel-space box to a normalized YOLO label line.

    Args:
        class_id: Index into the dataset's `classes.txt`.
        box: `(x_min, y_min, x_max, y_max)` pixel-space box.
        width: Image width, for normalization.
        height: Image height, for normalization.

    Returns:
        `"class_id cx cy w h"`, `cx`/`cy`/`w`/`h` normalized to `[0, 1]`.
    """
    x0, y0, x1, y1 = box
    cx = (x0 + x1) / 2 / width
    cy = (y0 + y1) / 2 / height
    w = (x1 - x0) / width
    h = (y1 - y0) / height
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def write_yolo_dataset(
    samples: list[Sample],
    source_images_dir: str | Path,
    object_class_name: str,
    defect_class_name: str,
    output_dir: str | Path,
) -> Path:
    """Write `samples` as a flat YOLO-format dataset under `output_dir`.

    Args:
        samples: Samples to include.
        source_images_dir: Directory each `Sample.image_path` is resolved against.
        object_class_name: `classes.txt`'s index-0 class.
        defect_class_name: `classes.txt`'s index-1 class.
        output_dir: Destination root -- gets `images/`, `labels/`, `classes.txt`.

    Returns:
        `output_dir`, as a `Path`.

    Raises:
        FileNotFoundError: If a sample's source image is missing.
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "classes.txt").write_text(f"{object_class_name}\n{defect_class_name}\n")

    for sample in samples:
        source_path = Path(source_images_dir) / sample.image_path
        if not source_path.is_file():
            raise FileNotFoundError(f"Sample image not found: {source_path}")
        dest_path = images_dir / Path(sample.image_path).name
        shutil.copy2(source_path, dest_path)

        lines = [to_yolo_line(0, sample.object_bbox, sample.width, sample.height)]
        lines += [to_yolo_line(1, box, sample.width, sample.height) for box in sample.defect_boxes]
        (labels_dir / f"{dest_path.stem}.txt").write_text("\n".join(lines) + "\n")

    return output_dir
