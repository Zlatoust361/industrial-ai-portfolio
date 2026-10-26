"""Thin `ultralytics.YOLO` inference wrapper (week 07).

Shared by `scripts/export_and_compare_jetson.py` (day 2) and
`scripts/compare_dr_branches.py` (day 4) -- both need to turn a loaded YOLO
model's raw output into this repo's `Prediction` dataclass the same way.
Lazily imports `ultralytics` inside each function, not at module level, so
importing this module costs nothing without it installed (not the case in
this sandbox -- see `README.md`'s week 07 section).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.evaluation.detection_metrics import Prediction


def load_yolo_model(model_path: str | Path) -> Any:
    """Load a YOLO model via `ultralytics.YOLO` -- a `.pt` checkpoint or a `.engine` file.

    Args:
        model_path: Path to the model file. `ultralytics` picks the right
            backend from the extension.

    Returns:
        A loaded `ultralytics.YOLO` instance.
    """
    from ultralytics import YOLO

    return YOLO(str(model_path))


def run_yolo_inference(
    model: Any, image_paths: list[Path], score_threshold: float
) -> list[Prediction]:
    """Run a loaded `ultralytics.YOLO` model over `image_paths`, collecting `Prediction`s.

    Args:
        model: A loaded `ultralytics.YOLO` instance (see `load_yolo_model`)
            -- either a `.pt` checkpoint (fp32) or a `.engine` TensorRT
            engine (fp16); both expose the same `.predict()` interface.
        image_paths: Images to run inference on.
        score_threshold: Minimum confidence a detection needs to be kept.

    Returns:
        One `Prediction` per kept detection, across all images. `class_name`
        comes from the model's own `.names` (its trained class map).
    """
    predictions: list[Prediction] = []
    results = model.predict(source=[str(p) for p in image_paths], verbose=False)
    for image_path, result in zip(image_paths, results, strict=True):
        for box in result.boxes:
            confidence = float(box.conf[0])
            if confidence < score_threshold:
                continue
            class_name = result.names[int(box.cls[0])]
            x0, y0, x1, y1 = (float(v) for v in box.xyxy[0])
            predictions.append(
                Prediction(
                    image_id=image_path.name,
                    class_name=class_name,
                    score=confidence,
                    box=(x0, y0, x1, y1),
                )
            )
    return predictions
