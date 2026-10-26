"""YOLOv8 (Ultralytics) wrapper, for compare_owlvit_yolov8.py only.

YOLOv8 is closed-set: it always predicts whatever classes the loaded
checkpoint was trained on, regardless of what's asked for -- unlike
`OwlViTMultiClassDetector`, `class_names` in `.detect()` is accepted only
to keep the two detectors' call signature uniform for the comparison
script's shared loop, and is otherwise unused.

**Caveat for a meaningful comparison**: the test set's `classes.txt` must
use the same class name strings the loaded YOLOv8 checkpoint predicts
(e.g. COCO class names for a stock `yolov8n.pt`) -- `mean_average_precision`
matches purely by string equality, so a name mismatch silently scores as
"no predictions for that class" rather than an error.

Not runnable in this sandbox: `ultralytics` isn't installed and there's no
network access to download YOLOv8 weights (see course/DECISIONS.md
ADR-004). Written from documented `ultralytics.YOLO` usage, not executed
here.
"""

from __future__ import annotations

import numpy as np

Box = tuple[float, float, float, float]


class YOLOv8Detector:
    """Runs a YOLOv8 checkpoint (`.pt`, or a pre-exported `.onnx`/`.engine`) via `ultralytics`."""

    def __init__(self, model_path: str, score_threshold: float = 0.1) -> None:
        """Args:
            model_path: Path (or Ultralytics Hub name) of the YOLOv8 checkpoint.
                `.pt` runs eager PyTorch; `.onnx`/`.engine` are auto-detected
                by `ultralytics` and run through their respective runtimes.
            score_threshold: Minimum detection confidence to report.
        """
        from ultralytics import YOLO

        self._model = YOLO(model_path)
        self._score_threshold = score_threshold

    def detect(self, image_rgb: np.ndarray, class_names: list[str]) -> list[tuple[str, float, Box]]:
        """Detect this checkpoint's own classes in `image_rgb`.

        Args:
            image_rgb: An `(height, width, 3)` uint8 RGB image array.
            class_names: Unused (see module docstring) -- kept only so this
                matches `OwlViTMultiClassDetector.detect`'s signature.

        Returns:
            `(class_name, score, box_xyxy)` tuples with `score >= score_threshold`.
        """
        del class_names  # unused, see module docstring
        results = self._model.predict(image_rgb, conf=self._score_threshold, verbose=False)[0]

        predictions: list[tuple[str, float, Box]] = []
        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = results.names[class_id]
            score = float(box.conf[0])
            x_min, y_min, x_max, y_max = (float(v) for v in box.xyxy[0])
            predictions.append((class_name, score, (x_min, y_min, x_max, y_max)))
        return predictions
