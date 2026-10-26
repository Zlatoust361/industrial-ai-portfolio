"""Unit tests for `classical_cv.detectors` -- real `cv2` calls, no mocking.

Thresholds below (`best IoU >= 0.2`, `0 false positives on a handful of
clean images`) are picked from this detector's own actually-measured
behaviour at `seed=42` (see `classical_cv/README.md`'s real numbers
table), not from a hoped-for target -- a regression here means the
detector's real behaviour changed, which is exactly what should fail.
"""

from __future__ import annotations

import numpy as np

from classical_cv.defect_images import build_demo_dataset, render_clean_image
from classical_cv.detectors import (
    DetectionCandidate,
    detect_blob_defects,
    detect_defects,
    detect_line_defects,
)
from src.evaluation.detection_metrics import compute_iou
from src.synthetic.config import load_defects_config

DEFECTS_CONFIG = load_defects_config("config/defects.yaml")


def test_detect_line_defects_returns_hough_line_candidates_only() -> None:
    dataset = build_demo_dataset(DEFECTS_CONFIG, num_positive=5, num_negative=0, seed=42)
    candidates = detect_line_defects(dataset[0].pixels)
    assert all(c.method == "hough_line" for c in candidates)
    assert all(0.0 <= c.score <= 1.0 for c in candidates)


def test_detect_blob_defects_returns_blob_contour_candidates_only() -> None:
    dataset = build_demo_dataset(DEFECTS_CONFIG, num_positive=5, num_negative=0, seed=42)
    candidates = detect_blob_defects(dataset[0].pixels)
    assert all(c.method == "blob_contour" for c in candidates)
    assert all(0.0 <= c.score <= 1.0 for c in candidates)


def test_detect_defects_finds_real_defects_with_reasonable_overlap() -> None:
    """Across a real batch of generated images, most ground truth defects
    should have at least one detection candidate overlapping them decently --
    not a perfect detector (classical CV boxes are coarse by construction),
    but not blind either."""
    dataset = build_demo_dataset(DEFECTS_CONFIG, num_positive=10, num_negative=0, seed=42)

    best_ious = []
    for image in dataset:
        candidates = detect_defects(image.pixels)
        for gt_box in image.ground_truth_boxes:
            best_iou = max((compute_iou(c.box, gt_box) for c in candidates), default=0.0)
            best_ious.append(best_iou)

    assert best_ious, "Expected at least one ground truth box across the batch."
    assert sum(iou >= 0.2 for iou in best_ious) / len(best_ious) >= 0.7


def test_detect_defects_has_few_false_positives_on_clean_images() -> None:
    rng = np.random.default_rng(100)
    clean_images = [render_clean_image(f"clean_{i}", rng) for i in range(10)]
    total_candidates = sum(len(detect_defects(image.pixels)) for image in clean_images)
    assert total_candidates <= 2  # measured: 0 at seed=42, small slack for other seeds


def test_detect_defects_applies_non_maximum_suppression() -> None:
    """Two near-identical overlapping boxes must collapse to one via NMS."""
    from classical_cv.detectors import NMS_IOU_THRESHOLD, _suppress_overlapping

    duplicate = DetectionCandidate(box=(10.0, 10.0, 50.0, 50.0), score=0.9, method="blob_contour")
    almost_duplicate = DetectionCandidate(
        box=(11.0, 11.0, 51.0, 51.0), score=0.5, method="hough_line"
    )
    distant = DetectionCandidate(box=(200.0, 200.0, 240.0, 240.0), score=0.4, method="blob_contour")

    kept = _suppress_overlapping([duplicate, almost_duplicate, distant], NMS_IOU_THRESHOLD)

    assert len(kept) == 2
    assert duplicate in kept
    assert distant in kept
    assert almost_duplicate not in kept
