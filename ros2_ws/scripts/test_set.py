"""Loads a YOLO-style test set: images/, labels/, classes.txt.

    test_set/
      images/*.jpg              # or .jpeg/.png
      labels/*.txt              # "class_id cx cy w h" per line, normalized [0,1],
                                 # one file per image (same stem, e.g. images/a.jpg -> labels/a.txt)
      classes.txt                # one class name per line, index = class_id

Only needs `PIL` (to read image dimensions, for normalized -> pixel-space
box conversion) -- no torch/model dependency, so it's unit-testable
without either.
"""

from __future__ import annotations

from pathlib import Path

from map_metric import GroundTruth

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_yolo_label_line(
    line: str,
    class_names: list[str],
    image_width: int,
    image_height: int,
    image_id: str,
) -> GroundTruth | None:
    """Parse one YOLO-format label line into a pixel-space `GroundTruth`.

    Args:
        line: `"class_id cx cy w h"`, with `cx`/`cy`/`w`/`h` normalized to `[0, 1]`.
        class_names: Index -> class name mapping.
        image_width: Image width in pixels (for denormalizing the box).
        image_height: Image height in pixels (for denormalizing the box).
        image_id: Identifier to stamp onto the resulting `GroundTruth`
            (typically the image's filename).

    Returns:
        A `GroundTruth` with a pixel-space `(x_min, y_min, x_max, y_max)`
        box, or `None` for a blank or `#`-comment line.

    Raises:
        ValueError: If the line doesn't have exactly 5 whitespace-separated
            fields, or `class_id` is out of range for `class_names`.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    parts = stripped.split()
    if len(parts) != 5:
        raise ValueError(f"Expected 'class_id cx cy w h', got: {line!r}")

    class_id = int(parts[0])
    if not (0 <= class_id < len(class_names)):
        raise ValueError(f"class_id {class_id} out of range for {len(class_names)} classes.")

    cx, cy, w, h = (float(part) for part in parts[1:])
    x_min = (cx - w / 2) * image_width
    y_min = (cy - h / 2) * image_height
    x_max = (cx + w / 2) * image_width
    y_max = (cy + h / 2) * image_height

    return GroundTruth(
        image_id=image_id, class_name=class_names[class_id], box=(x_min, y_min, x_max, y_max)
    )


def load_test_set(root: str | Path) -> tuple[list[Path], list[GroundTruth], list[str]]:
    """Load a YOLO-style test set from `root` (see module docstring for layout).

    Images without a matching `labels/<stem>.txt` are still returned in
    `image_paths` (detectors still run on them), just contribute no
    `GroundTruth` entries.

    Args:
        root: Test-set root directory.

    Returns:
        `(image_paths, ground_truths, class_names)`, `image_paths` sorted
        for reproducibility.

    Raises:
        FileNotFoundError: If `root`, `root/images`, or `root/classes.txt` is missing.
    """
    root = Path(root)
    images_dir = root / "images"
    labels_dir = root / "labels"
    classes_path = root / "classes.txt"

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Expected images directory not found: {images_dir}")
    if not classes_path.is_file():
        raise FileNotFoundError(f"Expected classes file not found: {classes_path}")

    class_lines = classes_path.read_text(encoding="utf-8").splitlines()
    class_names = [line.strip() for line in class_lines if line.strip()]

    from PIL import Image

    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)

    ground_truths: list[GroundTruth] = []
    for image_path in image_paths:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            continue

        with Image.open(image_path) as image:
            width, height = image.size

        for line in label_path.read_text(encoding="utf-8").splitlines():
            ground_truth = parse_yolo_label_line(
                line, class_names, width, height, image_id=image_path.name
            )
            if ground_truth is not None:
                ground_truths.append(ground_truth)

    return image_paths, ground_truths, class_names
