"""Measured accuracy/latency of the classical-CV detector (week 25).

Reuses `src.evaluation.detection_metrics`'s real, already-tested mAP/PR/F1
math (week 07) rather than reinventing scoring -- the exact same
`Prediction`/`GroundTruth`/`average_precision`/`precision_recall_f1_at_threshold`
`scripts/compare_dr_branches.py` uses to score YOLO. Latency is wall-clock
`time.perf_counter()` on whatever CPU this process actually runs on --
measured, not estimated, same standard the rest of the course holds
Jetson-latency claims to.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from classical_cv.defect_images import GeneratedImage
from classical_cv.detectors import DetectionCandidate, detect_defects
from src.evaluation.detection_metrics import (
    GroundTruth,
    Prediction,
    average_precision,
    precision_recall_f1_at_threshold,
)

DEFECT_CLASS_NAME = "defect"
"""Single class name classical CV scores against -- week 25 doesn't
distinguish `scratch`/`pit`/`crack` at detection time (none of the three
classical sub-detectors are shape-specific enough to claim that), only
"something was found here" vs. ground truth's "something real is here"."""

DEFAULT_IOU_THRESHOLD = 0.3
"""Lower than `src.evaluation.detection_metrics`'s own default of `0.5` --
classical contour/Hough boxes are coarse by construction (a `HoughLinesP`
segment padded by a few px, not a tightly regressed box), so demanding
YOLO-grade overlap would understate a detector that is, in fact, finding
the right defect.
"""

DEFAULT_SCORE_THRESHOLD = 0.2
"""Operating point used for the fixed-threshold precision/recall/F1 report
-- below this, both sub-detectors' heuristic scores are dominated by noise
(very short Hough segments / very small contours)."""


@dataclass(frozen=True)
class ImageBenchResult:
    """Per-image detector output, kept for the report's per-image table.

    Attributes:
        image_id: Matches `GeneratedImage.image_id`.
        candidates: What `detect_defects` found on this image.
        latency_ms: Wall-clock time `detect_defects` took on this image.
    """

    image_id: str
    candidates: tuple[DetectionCandidate, ...]
    latency_ms: float


@dataclass(frozen=True)
class ClassicalCVBenchReport:
    """Aggregate accuracy/latency over a whole dataset.

    Attributes:
        num_images: Images scored (positive + negative).
        num_positive: Images with at least one real ground truth defect.
        average_precision: AP at `iou_threshold`, see `average_precision`.
        operating_point: `precision_recall_f1_at_threshold`'s dict at
            `DEFAULT_SCORE_THRESHOLD`.
        mean_latency_ms: Mean of every image's `detect_defects` wall-clock time.
        p95_latency_ms: 95th percentile of the same per-image latencies.
        iou_threshold: IoU threshold `average_precision` was computed at.
        score_threshold: Score threshold `operating_point` was computed at.
        per_image: Every image's raw detections, for the markdown report.
    """

    num_images: int
    num_positive: int
    average_precision: float
    operating_point: dict[str, float]
    mean_latency_ms: float
    p95_latency_ms: float
    iou_threshold: float
    score_threshold: float
    per_image: tuple[ImageBenchResult, ...]


def _percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile -- no numpy dependency needed for this small a list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


def run_detector_on_dataset(images: list[GeneratedImage]) -> list[ImageBenchResult]:
    """Run `detect_defects` on every image, timing each call individually.

    Args:
        images: Dataset built by `classical_cv.defect_images.build_demo_dataset`.

    Returns:
        One `ImageBenchResult` per input image, same order.
    """
    results = []
    for image in images:
        start = time.perf_counter()
        candidates = detect_defects(image.pixels)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        results.append(
            ImageBenchResult(
                image_id=image.image_id, candidates=tuple(candidates), latency_ms=elapsed_ms
            )
        )
    return results


def _to_predictions(image_id: str, candidates: tuple[DetectionCandidate, ...]) -> list[Prediction]:
    return [
        Prediction(image_id=image_id, class_name=DEFECT_CLASS_NAME, score=c.score, box=c.box)
        for c in candidates
    ]


def _to_ground_truths(image: GeneratedImage) -> list[GroundTruth]:
    return [
        GroundTruth(image_id=image.image_id, class_name=DEFECT_CLASS_NAME, box=box)
        for box in image.ground_truth_boxes
    ]


def evaluate_classical_cv(
    images: list[GeneratedImage],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> ClassicalCVBenchReport:
    """Run and score the classical-CV detector over a full positive+negative dataset.

    Args:
        images: Dataset built by `classical_cv.defect_images.build_demo_dataset`
            (positives AND negatives -- negatives are what let this report
            honestly reflect false positives, see `detection_metrics.py`'s
            per-image matching).
        iou_threshold: Minimum IoU for a detection to count as a true positive.
        score_threshold: Confidence cutoff for the fixed-threshold operating point.

    Returns:
        A `ClassicalCVBenchReport` with real measured AP/precision/recall/F1/latency.
    """
    per_image = run_detector_on_dataset(images)

    predictions = [
        prediction
        for image_result in per_image
        for prediction in _to_predictions(image_result.image_id, image_result.candidates)
    ]
    ground_truths = [gt for image in images for gt in _to_ground_truths(image)]

    ap = average_precision(predictions, ground_truths, DEFECT_CLASS_NAME, iou_threshold)
    operating_point = precision_recall_f1_at_threshold(
        predictions, ground_truths, DEFECT_CLASS_NAME, score_threshold, iou_threshold
    )

    latencies = [result.latency_ms for result in per_image]
    num_positive = sum(1 for image in images if image.ground_truth_boxes)

    return ClassicalCVBenchReport(
        num_images=len(images),
        num_positive=num_positive,
        average_precision=ap,
        operating_point=operating_point,
        mean_latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
        p95_latency_ms=_percentile(latencies, 0.95),
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        per_image=tuple(per_image),
    )
