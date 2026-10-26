"""Turn raw OWL-ViT detections (score/box arrays) into `Detection` objects.

Deliberately doesn't reimplement OWL-ViT's box math (normalized cxcywh ->
pixel xyxy, letterbox correction): both OWL-ViT backends call HuggingFace's
own `OwlViTProcessor.post_process_object_detection()` for that -- it's
well-tested and identical between backends, so results are comparable
regardless of which one produced them. This module only does the last,
backend-agnostic step: already-pixel-space (score, box) pairs -> `Detection`.
Pure numpy/Python, no rclpy/torch/transformers needed, so it's unit-testable
without any of them installed.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from vision_bridge.detection import BoundingBox, Detection


def build_detections(
    scores: Iterable[float],
    boxes_xyxy: Iterable[Sequence[float]],
    label: str,
    score_threshold: float,
) -> list[Detection]:
    """Convert parallel score/box arrays into `Detection` objects above threshold.

    Every OWL-ViT call in this package uses exactly one text query, so every
    resulting detection shares the same `label` -- there's no per-detection
    label to read off HF's output (its `labels` field would just be an
    index into a single-element query list here).

    Args:
        scores: One confidence score per candidate box.
        boxes_xyxy: One `(x_min, y_min, x_max, y_max)` pixel-space box per
            candidate, same length/order as `scores`.
        label: The text query these detections matched.
        score_threshold: Minimum score to keep a detection. HF's
            `post_process_object_detection` typically already filters by
            threshold; re-checking here keeps this function correct
            standalone regardless of what the caller already did.

    Returns:
        Detections with `score >= score_threshold`, in input order.
    """
    detections: list[Detection] = []
    for score, box in zip(scores, boxes_xyxy, strict=True):
        if score < score_threshold:
            continue
        x_min, y_min, x_max, y_max = box
        detections.append(
            Detection(
                label=label,
                score=float(score),
                bbox=BoundingBox(
                    x_min=float(x_min), y_min=float(y_min), x_max=float(x_max), y_max=float(y_max)
                ),
            )
        )
    return detections
