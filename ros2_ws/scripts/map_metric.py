"""Pure mAP@IoU computation (Pascal VOC 2010+-style average precision).

Operates on plain (image_id, class_name, score, box_xyxy) predictions and
(image_id, class_name, box_xyxy) ground truths, so predictions from any
detector (OWL-ViT, YOLOv8, ...) score through the same code path. No
torch/model/dataset dependency -- unit-testable with small hand-built
examples (see test_map_metric.py).
"""

from __future__ import annotations

from dataclasses import dataclass

Box = tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max), pixel space


@dataclass(frozen=True)
class Prediction:
    """One detected box."""

    image_id: str
    class_name: str
    score: float
    box: Box


@dataclass(frozen=True)
class GroundTruth:
    """One ground-truth box."""

    image_id: str
    class_name: str
    box: Box


def compute_iou(box_a: Box, box_b: Box) -> float:
    """Intersection-over-union of two `(x_min, y_min, x_max, y_max)` boxes.

    Args:
        box_a: First box.
        box_b: Second box.

    Returns:
        IoU in `[0, 1]`, or `0.0` if the union area is zero.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area

    return inter_area / union if union > 0 else 0.0


def average_precision(
    predictions: list[Prediction],
    ground_truths: list[GroundTruth],
    class_name: str,
    iou_threshold: float,
) -> float:
    """Average precision for one class: greedy IoU matching + PR-curve integration.

    Standard detection-benchmark AP: predictions are sorted by score
    descending and greedily matched (highest IoU, not already claimed) to
    ground truths in the same image; a match with `IoU >= iou_threshold`
    is a true positive, otherwise a false positive. AP is the area under
    the *monotonic* precision envelope over recall (Pascal VOC 2010+
    "all-points interpolation" -- the same style Ultralytics' own YOLOv8
    metrics use), so results are comparable with familiar `mAP@0.5` numbers.

    Args:
        predictions: All predictions for `class_name`, any image, any order.
        ground_truths: All ground-truth boxes for `class_name`.
        class_name: The class being scored.
        iou_threshold: Minimum IoU for a prediction to count as a match
            (`0.5` for `mAP@0.5`).

    Returns:
        Average precision in `[0, 1]`. `0.0` if there are no ground truths
        for this class (nothing to recall) -- a class with predictions but
        no ground truth just can't score above 0 here.
    """
    class_preds = sorted(
        (p for p in predictions if p.class_name == class_name),
        key=lambda p: p.score,
        reverse=True,
    )
    class_gts = [gt for gt in ground_truths if gt.class_name == class_name]
    if not class_gts:
        return 0.0

    # Ground truths grouped by image; each entry tracks whether it's been claimed.
    gts_by_image: dict[str, list[list]] = {}
    for gt in class_gts:
        gts_by_image.setdefault(gt.image_id, []).append([gt.box, False])

    num_gt = len(class_gts)
    true_positive = [0] * len(class_preds)
    false_positive = [0] * len(class_preds)

    for i, pred in enumerate(class_preds):
        candidates = gts_by_image.get(pred.image_id, [])
        best_iou = 0.0
        best_index = -1
        for index, (gt_box, matched) in enumerate(candidates):
            if matched:
                continue
            iou = compute_iou(pred.box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_index = index

        if best_iou >= iou_threshold and best_index >= 0:
            candidates[best_index][1] = True
            true_positive[i] = 1
        else:
            false_positive[i] = 1

    cumulative_tp = _cumulative_sum(true_positive)
    cumulative_fp = _cumulative_sum(false_positive)
    recalls = [tp / num_gt for tp in cumulative_tp]
    precisions = [
        tp / (tp + fp) if (tp + fp) > 0 else 0.0
        for tp, fp in zip(cumulative_tp, cumulative_fp, strict=True)
    ]

    return _integrate_precision_recall(recalls, precisions)


def mean_average_precision(
    predictions: list[Prediction],
    ground_truths: list[GroundTruth],
    iou_threshold: float = 0.5,
) -> dict[str, float | dict[str, float]]:
    """`mAP@iou_threshold`, averaged over every class present in `ground_truths`.

    Args:
        predictions: All predictions, any class/image, any order.
        ground_truths: All ground-truth boxes.
        iou_threshold: Minimum IoU for a match (default `0.5`, i.e. `mAP@0.5`).

    Returns:
        `{"mAP": <float>, "per_class": {class_name: <float>, ...}}`.
        `"mAP"` is `0.0` if `ground_truths` is empty.
    """
    class_names = sorted({gt.class_name for gt in ground_truths})
    per_class = {
        class_name: average_precision(predictions, ground_truths, class_name, iou_threshold)
        for class_name in class_names
    }
    mean_ap = sum(per_class.values()) / len(per_class) if per_class else 0.0
    return {"mAP": mean_ap, "per_class": per_class}


def _cumulative_sum(values: list[int]) -> list[int]:
    """Running total of `values`, same length, e.g. `[1,0,1] -> [1,1,2]`."""
    total = 0
    result = []
    for value in values:
        total += value
        result.append(total)
    return result


def _integrate_precision_recall(recalls: list[float], precisions: list[float]) -> float:
    """Area under the monotonic precision envelope over recall (Pascal VOC 2010+ style).

    Args:
        recalls: Recall at each prediction rank, ascending prefix sums / num_gt.
        precisions: Precision at each prediction rank, same length as `recalls`.

    Returns:
        The integrated area, in `[0, 1]`.
    """
    padded_recall = [0.0, *recalls, 1.0]
    padded_precision = [0.0, *precisions, 0.0]

    # Make precision a monotonically non-increasing envelope, scanning right to left.
    for i in range(len(padded_precision) - 2, -1, -1):
        padded_precision[i] = max(padded_precision[i], padded_precision[i + 1])

    area = 0.0
    for i in range(1, len(padded_recall)):
        if padded_recall[i] != padded_recall[i - 1]:
            area += (padded_recall[i] - padded_recall[i - 1]) * padded_precision[i]
    return area
