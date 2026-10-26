"""Measured accuracy/latency for RT-DETR-tiny and YOLOv8n-scratch, same scoring as week 25.

Reuses `src.evaluation.detection_metrics` (week 07's real AP/precision/
recall math) at the STANDARD `iou_threshold=0.5` -- unlike week 25's
classical CV (which deliberately lowers it to 0.3 for its coarser
Hough/contour boxes, see `classical_cv/benchmark.py`), RT-DETR and YOLO
both regress boxes directly, so holding them to the same bar week 07's
own YOLO evaluation uses is the fair, honest comparison, not a lowered
bar to flatter either one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from modern_detectors.dataset import DEFECT_CLASS_NAME, ResizedImage
from src.evaluation.detection_metrics import (
    GroundTruth,
    Prediction,
    average_precision,
    precision_recall_f1_at_threshold,
)

DEFAULT_IOU_THRESHOLD = 0.5
DEFAULT_SCORE_THRESHOLD = 0.3


@dataclass(frozen=True)
class DetectorBenchReport:
    """Aggregate accuracy/latency for one trained detector.

    Attributes:
        detector_name: `"rt_detr_tiny"` or `"yolov8n_scratch"`.
        num_images: Images scored.
        average_precision: AP at `iou_threshold`.
        operating_point: `precision_recall_f1_at_threshold`'s dict at
            `score_threshold`.
        mean_latency_ms: Mean per-image inference wall-clock time.
        iou_threshold: IoU threshold `average_precision` was computed at.
        score_threshold: Score threshold `operating_point` was computed at.
    """

    detector_name: str
    num_images: int
    average_precision: float
    operating_point: dict[str, float]
    mean_latency_ms: float
    iou_threshold: float
    score_threshold: float


def score_predictions(
    detector_name: str,
    predictions_by_image: dict[str, list[Prediction]],
    images: list[ResizedImage],
    latencies_ms: list[float],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> DetectorBenchReport:
    """Score one detector's predictions against `images`' real ground truth.

    Args:
        detector_name: Label for the report (not used in scoring itself).
        predictions_by_image: `image_id -> list[Prediction]`, as returned
            by `modern_detectors.rt_detr_scratch.predict_rt_detr` or
            `modern_detectors.yolo_scratch.predict_yolo`.
        images: The same images the predictions were made on (for ground truth).
        latencies_ms: Per-image inference time, same order as `images`.
        iou_threshold: Minimum IoU for a true positive.
        score_threshold: Confidence cutoff for the fixed-threshold operating point.

    Returns:
        A `DetectorBenchReport` with real measured numbers.
    """
    predictions = [p for preds in predictions_by_image.values() for p in preds]
    ground_truths = [
        GroundTruth(image_id=image.image_id, class_name=DEFECT_CLASS_NAME, box=box)
        for image in images
        for box in image.ground_truth_boxes
    ]

    ap = average_precision(predictions, ground_truths, DEFECT_CLASS_NAME, iou_threshold)
    operating_point = precision_recall_f1_at_threshold(
        predictions, ground_truths, DEFECT_CLASS_NAME, score_threshold, iou_threshold
    )

    return DetectorBenchReport(
        detector_name=detector_name,
        num_images=len(images),
        average_precision=ap,
        operating_point=operating_point,
        mean_latency_ms=sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
    )


def measure_per_image_latency_ms(predict_one, images: list[ResizedImage]) -> list[float]:
    """Time `predict_one(image)` for every image individually.

    Args:
        predict_one: A zero-argument-per-call callable, `image -> anything`
            (return value discarded -- this is purely for timing a single
            image's worth of inference, as a deployed system would see it).
        images: Images to time inference on.

    Returns:
        Wall-clock milliseconds per image, same order as `images`.
    """
    latencies = []
    for image in images:
        start = time.perf_counter()
        predict_one(image)
        latencies.append((time.perf_counter() - start) * 1000.0)
    return latencies
