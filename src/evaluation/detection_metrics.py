"""Object-detection metrics: mAP@IoU, PR/ROC curves, threshold-point P/R/F1,
plus a YOLO-format test-set loader and a `Prediction` JSON loader.

Pure IoU/AP/ROC math plus `PIL`/`json`-only I/O -- no `torch`/`ultralytics`
import, so this is unit-testable without either installed. Used by
`scripts/export_and_compare_jetson.py` and `scripts/compare_dr_branches.py`
(week 07) to score a YOLO model's predictions against ground truth; the
mAP algorithm (greedy per-image IoU matching, Pascal VOC 2010+-style
monotonic precision-envelope integration) is the same one
`ros2_ws/scripts/map_metric.py` (week 03) uses, reimplemented fresh here
rather than imported from there -- `ros2_ws/` is a separate colcon
workspace by design (see `ros2_ws/README.md`), not a package this repo's
root code depends on.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

Box = tuple[float, float, float, float]
"""`(x_min, y_min, x_max, y_max)` in pixel coordinates."""

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class Prediction:
    """One detector output.

    Attributes:
        image_id: Identifies which image this came from (matched against
            `GroundTruth.image_id`, e.g. the filename).
        class_name: Predicted class label.
        score: Confidence score; higher ranks first when computing AP.
        box: Predicted bounding box.
    """

    image_id: str
    class_name: str
    score: float
    box: Box


@dataclass(frozen=True)
class GroundTruth:
    """One ground-truth annotation.

    Attributes:
        image_id: Identifies which image this belongs to.
        class_name: True class label.
        box: Ground-truth bounding box.
    """

    image_id: str
    class_name: str
    box: Box


def compute_iou(box_a: Box, box_b: Box) -> float:
    """Intersection-over-union of two axis-aligned boxes.

    Args:
        box_a: First box.
        box_b: Second box.

    Returns:
        IoU in `[0, 1]`. `0.0` if the boxes don't overlap, or either has zero area.
    """
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b

    inter_x0, inter_y0 = max(ax0, bx0), max(ay0, by0)
    inter_x1, inter_y1 = min(ax1, bx1), min(ay1, by1)
    inter_area = max(0.0, inter_x1 - inter_x0) * max(0.0, inter_y1 - inter_y0)

    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union_area = area_a + area_b - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def _cumulative_sum(values: list[int]) -> list[int]:
    """Running total of `values`, same length."""
    total = 0
    result = []
    for value in values:
        total += value
        result.append(total)
    return result


def _match_class_predictions(
    predictions: list[Prediction],
    ground_truths: list[GroundTruth],
    class_name: str,
    iou_threshold: float,
) -> tuple[list[int], list[int], int]:
    """Greedy highest-IoU-first matching for one class, ranked by score descending.

    Shared by `average_precision` and `precision_recall_curve` -- both need
    the exact same rank-ordered true/false-positive assignment, just
    integrated differently.

    Returns:
        `(true_positive_flags, false_positive_flags, num_ground_truths)`,
        one flag per ranked prediction (highest score first).
    """
    class_ground_truths = [gt for gt in ground_truths if gt.class_name == class_name]
    class_predictions = sorted(
        (p for p in predictions if p.class_name == class_name), key=lambda p: p.score, reverse=True
    )

    gt_by_image: dict[str, list[GroundTruth]] = {}
    for gt in class_ground_truths:
        gt_by_image.setdefault(gt.image_id, []).append(gt)
    matched: dict[int, bool] = {id(gt): False for gt in class_ground_truths}

    true_positives: list[int] = []
    false_positives: list[int] = []
    for prediction in class_predictions:
        candidates = gt_by_image.get(prediction.image_id, [])
        best_iou, best_gt = 0.0, None
        for gt in candidates:
            if matched[id(gt)]:
                continue
            iou = compute_iou(prediction.box, gt.box)
            if iou > best_iou:
                best_iou, best_gt = iou, gt

        if best_gt is not None and best_iou >= iou_threshold:
            matched[id(best_gt)] = True
            true_positives.append(1)
            false_positives.append(0)
        else:
            true_positives.append(0)
            false_positives.append(1)

    return true_positives, false_positives, len(class_ground_truths)


def _cumulative_recall_precision(
    true_positives: list[int], false_positives: list[int], num_ground_truths: int
) -> tuple[list[float], list[float]]:
    """Cumulative (recall, precision) at each rank, from per-rank TP/FP flags."""
    cumulative_tp = _cumulative_sum(true_positives)
    cumulative_fp = _cumulative_sum(false_positives)
    recalls = [tp / num_ground_truths for tp in cumulative_tp]
    precisions = [
        tp / (tp + fp) if (tp + fp) > 0 else 0.0
        for tp, fp in zip(cumulative_tp, cumulative_fp, strict=True)
    ]
    return recalls, precisions


def _integrate_precision_recall(recalls: list[float], precisions: list[float]) -> float:
    """Area under the monotonic (non-increasing) precision envelope, Pascal VOC 2010+-style.

    Args:
        recalls: Cumulative recall at each rank, ascending.
        precisions: Cumulative precision at each rank, same order.

    Returns:
        Average precision, in `[0, 1]`.
    """
    padded_recalls = [0.0, *recalls, 1.0]
    padded_precisions = [0.0, *precisions, 0.0]

    for i in range(len(padded_precisions) - 2, -1, -1):
        padded_precisions[i] = max(padded_precisions[i], padded_precisions[i + 1])

    area = 0.0
    for i in range(1, len(padded_recalls)):
        area += (padded_recalls[i] - padded_recalls[i - 1]) * padded_precisions[i]
    return area


def average_precision(
    predictions: list[Prediction],
    ground_truths: list[GroundTruth],
    class_name: str,
    iou_threshold: float = 0.5,
) -> float:
    """Average precision for one class, greedy highest-IoU-first matching per image.

    Args:
        predictions: All predictions (any class -- filtered internally).
        ground_truths: All ground truths (any class -- filtered internally).
        class_name: Class to score.
        iou_threshold: Minimum IoU for a prediction to count as a true positive.

    Returns:
        AP in `[0, 1]`. `0.0` if there are no ground truths for `class_name`.
    """
    true_positives, false_positives, num_gt = _match_class_predictions(
        predictions, ground_truths, class_name, iou_threshold
    )
    if num_gt == 0:
        return 0.0
    recalls, precisions = _cumulative_recall_precision(true_positives, false_positives, num_gt)
    return _integrate_precision_recall(recalls, precisions)


def mean_average_precision(
    predictions: list[Prediction], ground_truths: list[GroundTruth], iou_threshold: float = 0.5
) -> dict[str, object]:
    """Mean AP across every class present in `ground_truths`.

    Args:
        predictions: All predictions.
        ground_truths: All ground truths.
        iou_threshold: Minimum IoU for a true positive.

    Returns:
        `{"mAP": float, "per_class": {class_name: ap}}`. `mAP` is `0.0` for
        an empty `ground_truths`.
    """
    class_names = sorted({gt.class_name for gt in ground_truths})
    if not class_names:
        return {"mAP": 0.0, "per_class": {}}

    per_class = {
        class_name: average_precision(predictions, ground_truths, class_name, iou_threshold)
        for class_name in class_names
    }
    return {"mAP": sum(per_class.values()) / len(per_class), "per_class": per_class}


def precision_recall_curve(
    predictions: list[Prediction],
    ground_truths: list[GroundTruth],
    class_name: str,
    iou_threshold: float = 0.5,
) -> tuple[list[float], list[float]]:
    """Raw (recall, precision) points at each ranked prediction, for plotting.

    Unlike `average_precision`'s internal use of the monotonic precision
    envelope (a Pascal-VOC-style integration trick), these are the actual
    cumulative values at each rank -- the usual zigzag PR curve shape, not
    smoothed.

    Args:
        predictions: All predictions (any class -- filtered internally).
        ground_truths: All ground truths (any class -- filtered internally).
        class_name: Class to score.
        iou_threshold: Minimum IoU for a true positive.

    Returns:
        `(recalls, precisions)`, same length, ascending by rank (score
        descending). Both empty if there are no ground truths for `class_name`.
    """
    true_positives, false_positives, num_gt = _match_class_predictions(
        predictions, ground_truths, class_name, iou_threshold
    )
    if num_gt == 0:
        return [], []
    return _cumulative_recall_precision(true_positives, false_positives, num_gt)


def precision_recall_f1_at_threshold(
    predictions: list[Prediction],
    ground_truths: list[GroundTruth],
    class_name: str,
    score_threshold: float,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Precision/recall/F1 at one fixed confidence threshold -- an operating point, not a curve.

    Unlike `average_precision` (which sweeps every possible threshold),
    this reports the single point a detector would actually run at in
    production: only predictions scoring `>= score_threshold` are
    considered at all (still matched in ranked order, so a higher-confidence
    prediction claims a shared ground truth first).

    Args:
        predictions: All predictions (any class -- filtered internally).
        ground_truths: All ground truths (any class -- filtered internally).
        class_name: Class to score.
        score_threshold: Minimum confidence to keep a prediction at all.
        iou_threshold: Minimum IoU for a true positive.

    Returns:
        `{"precision", "recall", "f1", "true_positives", "false_positives",
        "false_negatives"}`. `precision`/`recall`/`f1` are `0.0` when their
        denominator is `0`.
    """
    thresholded = [p for p in predictions if p.score >= score_threshold]
    true_positives, false_positives, num_gt = _match_class_predictions(
        thresholded, ground_truths, class_name, iou_threshold
    )

    tp, fp = sum(true_positives), sum(false_positives)
    fn = num_gt - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }


def per_image_f1_scores(
    predictions: list[Prediction],
    ground_truths: list[GroundTruth],
    image_ids: list[str],
    class_name: str,
    score_threshold: float,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """F1 for `class_name`, computed independently per image.

    Each image is treated as its own tiny detection problem (its own
    predictions/ground-truths only) -- this is what makes a *paired*
    statistical test (see `src.evaluation.statistics.paired_t_test`)
    possible between two branches: both are scored on the exact same set
    of images, so the per-image F1 arrays line up index-for-index.

    Args:
        predictions: All predictions (any image/class -- filtered internally).
        ground_truths: All ground truths (any image/class -- filtered internally).
        image_ids: Every image to score.
        class_name: Class to score.
        score_threshold: Minimum confidence to keep a prediction at all.
        iou_threshold: Minimum IoU for a true positive.

    Returns:
        `image_id -> F1`, one entry per `image_ids` entry. An image with no
        ground truth and no (kept) prediction for `class_name` scores `1.0`
        (correctly predicting "nothing here"), matching
        `precision_recall_f1_at_threshold`'s `0/0 -> 0.0` convention would
        instead be undefined -- but per-image, "no GT, no prediction" is a
        true negative, not an ambiguous case.
    """
    scores: dict[str, float] = {}
    for image_id in image_ids:
        image_predictions = [
            p for p in predictions if p.image_id == image_id and p.score >= score_threshold
        ]
        image_ground_truths = [gt for gt in ground_truths if gt.image_id == image_id]
        if not image_predictions and not image_ground_truths:
            scores[image_id] = 1.0
            continue
        result = precision_recall_f1_at_threshold(
            image_predictions, image_ground_truths, class_name, score_threshold, iou_threshold
        )
        scores[image_id] = result["f1"]
    return scores


def image_level_labels_and_scores(
    predictions: list[Prediction],
    ground_truths: list[GroundTruth],
    image_ids: list[str],
    class_name: str,
) -> tuple[list[int], list[float]]:
    """Reduce per-box detection to a per-image binary "is `class_name` present" signal.

    Object detection doesn't have a natural ROC curve (no fixed population
    of "negative boxes" to compute a false-positive *rate* against) -- this
    reframes it as one binary classification problem per image: does the
    image contain `class_name` at all (ground truth), and what's the
    detector's strongest signal that it does (max predicted score for
    `class_name` on that image, `0.0` if none predicted)? That DOES have a
    well-defined ROC curve.

    Args:
        predictions: All predictions (any class/image -- filtered internally).
        ground_truths: All ground truths (any class/image -- filtered internally).
        image_ids: Every image to score (defines the ROC curve's population
            -- typically every image in the val set, not just ones with a detection).
        class_name: Class to score.

    Returns:
        `(labels, scores)`, one entry per `image_ids` entry, same order.
        `labels[i]` is `1` if that image has `>= 1` ground-truth box of
        `class_name`, else `0`.
    """
    positive_images = {gt.image_id for gt in ground_truths if gt.class_name == class_name}
    best_score_by_image: dict[str, float] = {}
    for prediction in predictions:
        if prediction.class_name != class_name:
            continue
        best_score_by_image[prediction.image_id] = max(
            best_score_by_image.get(prediction.image_id, 0.0), prediction.score
        )

    labels = [1 if image_id in positive_images else 0 for image_id in image_ids]
    scores = [best_score_by_image.get(image_id, 0.0) for image_id in image_ids]
    return labels, scores


def roc_curve(
    labels: list[int], scores: list[float]
) -> tuple[list[float], list[float], list[float]]:
    """Receiver operating characteristic curve for a binary classifier.

    Args:
        labels: True binary labels (`0`/`1`), one per sample.
        scores: Predicted scores (higher means more likely positive), same order.

    Returns:
        `(fpr, tpr, thresholds)`, ascending by `fpr`, starting at `(0, 0)`
        (nothing classified positive) and ending at `(1, 1)` (everything
        classified positive).

    Raises:
        ValueError: If `labels` has no positive or no negative samples (an
            ROC curve isn't defined without both).
    """
    num_positive = sum(labels)
    num_negative = len(labels) - num_positive
    if num_positive == 0 or num_negative == 0:
        raise ValueError(
            "roc_curve needs at least one positive and one negative label, got "
            f"{num_positive} positive / {num_negative} negative."
        )

    ranked = sorted(zip(scores, labels, strict=True), key=lambda pair: pair[0], reverse=True)

    fpr, tpr, thresholds = [0.0], [0.0], [math.inf]
    true_positive_count = false_positive_count = 0
    for score, label in ranked:
        if label == 1:
            true_positive_count += 1
        else:
            false_positive_count += 1
        tpr.append(true_positive_count / num_positive)
        fpr.append(false_positive_count / num_negative)
        thresholds.append(score)

    return fpr, tpr, thresholds


def roc_auc_score(fpr: list[float], tpr: list[float]) -> float:
    """Area under an ROC curve, via the trapezoidal rule.

    Args:
        fpr: False-positive rates, ascending (as returned by `roc_curve`).
        tpr: True-positive rates, same order.

    Returns:
        AUC in `[0, 1]`.
    """
    area = 0.0
    for i in range(1, len(fpr)):
        area += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2
    return area


def parse_yolo_label_line(
    line: str,
    class_names: list[str],
    image_width: int,
    image_height: int,
    image_id: str,
) -> GroundTruth | None:
    """Parse one YOLO-format label line into a pixel-space `GroundTruth`.

    Args:
        line: `"class_id cx cy w h"`, `cx`/`cy`/`w`/`h` normalized to `[0, 1]`.
        class_names: Index -> class name mapping.
        image_width: Image width in pixels (for denormalizing the box).
        image_height: Image height in pixels (for denormalizing the box).
        image_id: Identifier stamped onto the resulting `GroundTruth`
            (typically the image's filename).

    Returns:
        A `GroundTruth`, or `None` for a blank or `#`-comment line.

    Raises:
        ValueError: If the line doesn't have exactly 5 fields, or `class_id`
            is out of range for `class_names`.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    parts = stripped.split()
    if len(parts) != 5:
        raise ValueError(f"Expected 'class_id cx cy w h', got: {line!r}")

    class_id = int(parts[0])
    if not 0 <= class_id < len(class_names):
        raise ValueError(f"class_id {class_id} out of range for {len(class_names)} classes.")

    cx, cy, w, h = (float(part) for part in parts[1:])
    return GroundTruth(
        image_id=image_id,
        class_name=class_names[class_id],
        box=(
            (cx - w / 2) * image_width,
            (cy - h / 2) * image_height,
            (cx + w / 2) * image_width,
            (cy + h / 2) * image_height,
        ),
    )


def load_yolo_test_set(root: str | Path) -> tuple[list[Path], list[GroundTruth], list[str]]:
    """Load a YOLO-style test set: `root/images/*`, `root/labels/*.txt`, `root/classes.txt`.

    Images without a matching `labels/<stem>.txt` are still returned in
    `image_paths` (a detector still runs on them), just contribute no
    `GroundTruth` entries.

    Args:
        root: Test-set root directory.

    Returns:
        `(image_paths, ground_truths, class_names)`, `image_paths` sorted
        for reproducibility.

    Raises:
        FileNotFoundError: If `root/images` or `root/classes.txt` is missing.
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


def load_predictions_json(path: str | Path) -> list[Prediction]:
    """Load predictions from a JSON file.

    Expected shape: `{"detections": [{"image_id", "class_name", "score",
    "box": [x_min, y_min, x_max, y_max]}, ...]}`. NOT the same shape
    `pipeline/scripts/04_detect.py` (week 04) writes to `outputs/detections.json`
    (that one nests boxes per-image with a `"confidence"` key and no
    coordinates, since the quality-gate pipeline never needed per-box
    geometry) -- this is a standalone format for this module's own
    per-branch precomputed-detections demo path (see
    `scripts/compare_dr_branches.py`).

    Args:
        path: Path to the JSON file.

    Returns:
        One `Prediction` per `"detections"` entry.

    Raises:
        FileNotFoundError: If `path` doesn't exist.
        KeyError: If an entry is missing `image_id`/`class_name`/`score`/`box`.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Predictions file not found: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        Prediction(
            image_id=entry["image_id"],
            class_name=entry["class_name"],
            score=float(entry["score"]),
            box=tuple(entry["box"]),
        )
        for entry in raw["detections"]
    ]
