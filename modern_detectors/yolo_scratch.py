"""A real, from-scratch (no pretrained weights) YOLOv8n, trained on the same data as
`rt_detr_scratch.py` -- the fair, apples-to-apples half of week 26's comparison.

`YOLO("yolov8n.yaml")` builds the real `ultralytics` YOLOv8n architecture
from its bundled config file -- random weights, no network call (unlike
`scripts/train_dr_yolo.py`'s week 06/07 `YOLO("yolov8n.pt")`, which
fine-tunes a COCO-pretrained checkpoint downloaded on first use). Week 26
deliberately trains BOTH architectures from scratch on the identical
dataset so the comparison isolates architecture family (anchor-based CNN
vs. anchor-free transformer) from "one of them got a pretrained head
start" -- a fine-tuned-vs-scratch comparison would answer a different
question than the one this week asks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from ultralytics import YOLO

from classical_cv.defect_images import Box
from modern_detectors.dataset import TRAIN_IMAGE_SIZE, ResizedImage, write_yolo_dataset
from src.evaluation.detection_metrics import Prediction

YOLO_ARCHITECTURE = "yolov8n.yaml"
"""Architecture-only config bundled with `ultralytics` -- no pretrained
checkpoint download, see the module docstring."""


@dataclass(frozen=True)
class YoloTrainingRun:
    """Outcome of `train_yolo_from_scratch`.

    Attributes:
        model: The trained `YOLO` wrapper, ready for `predict_yolo`.
        train_seconds: Real wall-clock training time.
    """

    model: YOLO
    train_seconds: float


def train_yolo_from_scratch(
    train_images: list[ResizedImage],
    dataset_dir: str | Path,
    epochs: int,
    batch_size: int,
    seed: int,
) -> YoloTrainingRun:
    """Write `train_images` as a YOLO dataset and train YOLOv8n on it from scratch.

    Args:
        train_images: Training images (see `modern_detectors.dataset.resize_dataset`).
        dataset_dir: Directory to write the YOLO-format dataset into.
        epochs: Training epochs.
        batch_size: Images per batch.
        seed: `ultralytics`' own training seed (dataloader shuffling, augmentation).

    Returns:
        A `YoloTrainingRun` with the trained model and real elapsed time.
    """
    data_yaml_path = write_yolo_dataset(train_images, dataset_dir)
    model = YOLO(YOLO_ARCHITECTURE)

    start = time.perf_counter()
    model.train(
        data=str(data_yaml_path),
        epochs=epochs,
        imgsz=TRAIN_IMAGE_SIZE,
        batch=batch_size,
        device="cpu",
        workers=0,
        plots=False,
        verbose=False,
        exist_ok=True,
        seed=seed,
        project=str(Path(dataset_dir) / "runs"),
    )
    elapsed = time.perf_counter() - start

    return YoloTrainingRun(model=model, train_seconds=elapsed)


MIN_SCORE_FOR_SCORING = 0.001
"""`model.predict`'s own `conf` default (`0.25`) silently drops every box
below it BEFORE this week's scoring code ever sees it -- for a toy,
few-epoch from-scratch model, real confidences can legitimately sit at
~0.01 while the ranking among them is still meaningful (caught
empirically: AP read a flat `0.0` until this was lowered, see
`modern_detectors/README.md`'s Honesty note). `average_precision`/
`precision_recall_f1_at_threshold` do their OWN thresholding afterward,
the same way `rt_detr_scratch.predict_rt_detr`'s `score_threshold=0.0`
keeps every query for `benchmark.py` to threshold instead."""


def predict_yolo(model: YOLO, images: list[ResizedImage]) -> dict[str, list[Prediction]]:
    """Run inference, one image at a time (so each gets its own real latency measurement).

    Args:
        model: A trained `YOLO` wrapper from `train_yolo_from_scratch`.
        images: Images to run inference on.

    Returns:
        `image_id -> list[Prediction]`, in `TRAIN_IMAGE_SIZE` pixel space.
    """
    predictions: dict[str, list[Prediction]] = {}
    for image in images:
        results = model.predict(
            image.rgb_pixels, imgsz=TRAIN_IMAGE_SIZE, verbose=False, conf=MIN_SCORE_FOR_SCORING
        )
        boxes: list[Prediction] = []
        for result in results:
            for box, score in zip(result.boxes.xyxy, result.boxes.conf, strict=True):
                x0, y0, x1, y1 = (float(v) for v in box)
                box_t: Box = (x0, y0, x1, y1)
                boxes.append(
                    Prediction(
                        image_id=image.image_id, class_name="defect", score=float(score), box=box_t
                    )
                )
        predictions[image.image_id] = boxes
    return predictions
