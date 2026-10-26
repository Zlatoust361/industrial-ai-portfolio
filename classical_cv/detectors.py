"""Classical CV defect detection: Canny+Hough for lines, threshold+contours for blobs.

No neural network anywhere in this module -- real `cv2` calls only, the
whole point of week 25 being a genuine non-VLM baseline to compare against
(`src.models.zero_shot`'s cosine-similarity approach, weeks 01-02), not a
reimplementation that secretly still needs a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from classical_cv.defect_images import Box

CANNY_LOW_THRESHOLD = 40
CANNY_HIGH_THRESHOLD = 120
"""2:1-3:1 low:high ratio -- the usual OpenCV-docs rule of thumb, not tuned
per-image (this week's point is a generic baseline, not a hand-fit one)."""

HOUGH_MIN_LINE_LENGTH = 25
HOUGH_MAX_LINE_GAP = 6
"""Fed to `cv2.HoughLinesP` -- short enough to still catch `pit_range`-scale
scratches (`config/defects.yaml`'s `scratch.length_range` starts at 40px),
small enough gap that a `crack`'s bent polyline segments still link up."""

BLOB_DARKNESS_MARGIN = 35
"""A pixel counts as part of a blob candidate once it's at least this much
darker than the image's own median -- comfortably below
`defect_images.DEFECT_INTENSITY_DROP` (95) so real noise doesn't erase it,
comfortably above `BACKGROUND_NOISE_STD` (6) so background noise alone
doesn't trip it."""

MIN_BLOB_AREA_PX = 40
"""Drop contours smaller than this -- noise speckles, not real defects."""

NMS_IOU_THRESHOLD = 0.3
"""Two candidate boxes overlapping at least this much are treated as the
same defect; the lower-score one is dropped (same `compute_iou` helper
week 25 reuses for scoring, used here for suppression instead)."""


@dataclass(frozen=True)
class DetectionCandidate:
    """One classical-CV detection.

    Attributes:
        box: Pixel bounding box, `src.evaluation.detection_metrics.Box` convention.
        score: Heuristic confidence in `[0, 1]`, see `detect_defects`'s docstring.
        method: `"hough_line"` or `"blob_contour"` -- which sub-detector produced it.
    """

    box: Box
    score: float
    method: str


def _box_from_points(points: list[tuple[float, float]], pad: float) -> Box:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)


def detect_line_defects(image: np.ndarray) -> list[DetectionCandidate]:
    """Find scratch/crack-like line defects via Canny edges + probabilistic Hough transform.

    Args:
        image: `(height, width)` `uint8` grayscale array.

    Returns:
        One `DetectionCandidate` per detected line segment, `method="hough_line"`,
        `score` = segment length normalized by the image diagonal (longer
        segments read as more confident detections, capped at `1.0`).
    """
    blurred = cv2.GaussianBlur(image, ksize=(3, 3), sigmaX=0)
    edges = cv2.Canny(blurred, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)
    segments = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=20,
        minLineLength=HOUGH_MIN_LINE_LENGTH,
        maxLineGap=HOUGH_MAX_LINE_GAP,
    )
    if segments is None:
        return []

    height, width = image.shape[:2]
    diagonal = float(np.hypot(width, height))
    candidates = []
    for segment in segments:
        x0, y0, x1, y1 = (float(v) for v in segment[0])
        length = float(np.hypot(x1 - x0, y1 - y0))
        box = _box_from_points([(x0, y0), (x1, y1)], pad=4.0)
        score = min(1.0, length / diagonal * 4.0)
        candidates.append(DetectionCandidate(box=box, score=score, method="hough_line"))
    return candidates


def detect_blob_defects(image: np.ndarray) -> list[DetectionCandidate]:
    """Find pit-like blob defects via darkness thresholding + morphology + contours.

    Args:
        image: `(height, width)` `uint8` grayscale array.

    Returns:
        One `DetectionCandidate` per surviving contour, `method="blob_contour"`,
        `score` = contour area normalized by the image area (capped at `1.0`).
    """
    median_intensity = float(np.median(image))
    threshold_value = max(0, median_intensity - BLOB_DARKNESS_MARGIN)
    _, binary = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = image.shape[:2]
    image_area = float(width * height)

    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_BLOB_AREA_PX:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        score = min(1.0, (area / image_area) * 200.0)
        candidates.append(
            DetectionCandidate(
                box=(float(x), float(y), float(x + w), float(y + h)),
                score=score,
                method="blob_contour",
            )
        )
    return candidates


def _iou(box_a: Box, box_b: Box) -> float:
    """Local re-derivation of `src.evaluation.detection_metrics.compute_iou`'s formula.

    Imported lazily inside the function (not at module level) to avoid a
    hard import-time dependency of this detector module on the evaluation
    module -- `benchmark.py` already imports both directly for scoring, so
    this keeps `detectors.py` usable standalone (e.g. a quick manual
    `python -c` check) without needing `src.evaluation` on the path.
    """
    from src.evaluation.detection_metrics import compute_iou

    return compute_iou(box_a, box_b)


def _suppress_overlapping(
    candidates: list[DetectionCandidate], iou_threshold: float
) -> list[DetectionCandidate]:
    """Greedy highest-score-first non-maximum suppression."""
    ranked = sorted(candidates, key=lambda c: c.score, reverse=True)
    kept: list[DetectionCandidate] = []
    for candidate in ranked:
        if all(_iou(candidate.box, other.box) < iou_threshold for other in kept):
            kept.append(candidate)
    return kept


def detect_defects(image: np.ndarray) -> list[DetectionCandidate]:
    """Run both classical sub-detectors and merge overlapping candidates.

    Args:
        image: `(height, width)` `uint8` grayscale array.

    Returns:
        Non-maximum-suppressed candidates from `detect_line_defects` and
        `detect_blob_defects` combined, highest score first.
    """
    candidates = detect_line_defects(image) + detect_blob_defects(image)
    return _suppress_overlapping(candidates, NMS_IOU_THRESHOLD)
