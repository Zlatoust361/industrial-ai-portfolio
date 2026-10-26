"""Unit tests for map_metric.py.

No dependencies beyond stdlib + pytest. Run with:
    cd ros2_ws/scripts && pytest test_map_metric.py
"""

import pytest
from map_metric import (
    GroundTruth,
    Prediction,
    average_precision,
    compute_iou,
    mean_average_precision,
)


def test_compute_iou_identical_boxes_is_one() -> None:
    box = (0.0, 0.0, 10.0, 10.0)
    assert compute_iou(box, box) == 1.0


def test_compute_iou_disjoint_boxes_is_zero() -> None:
    assert compute_iou((0, 0, 1, 1), (5, 5, 6, 6)) == 0.0


def test_compute_iou_partial_overlap() -> None:
    # box_a area 4, box_b area 4, intersection area 1 -> union 7.
    assert compute_iou((0, 0, 2, 2), (1, 1, 3, 3)) == pytest.approx(1 / 7)


def test_compute_iou_zero_area_box_is_zero() -> None:
    assert compute_iou((0, 0, 0, 0), (0, 0, 10, 10)) == 0.0


def test_average_precision_perfect_predictions_is_one() -> None:
    gts = [GroundTruth(image_id=f"img{i}", class_name="cat", box=(0, 0, 10, 10)) for i in range(3)]
    preds = [
        Prediction(image_id=f"img{i}", class_name="cat", score=0.9 - i * 0.1, box=(0, 0, 10, 10))
        for i in range(3)
    ]
    assert average_precision(preds, gts, "cat", iou_threshold=0.5) == pytest.approx(1.0)


def test_average_precision_no_predictions_is_zero() -> None:
    gts = [GroundTruth(image_id="img0", class_name="cat", box=(0, 0, 10, 10))]
    assert average_precision([], gts, "cat", iou_threshold=0.5) == 0.0


def test_average_precision_no_ground_truth_is_zero() -> None:
    preds = [Prediction(image_id="img0", class_name="cat", score=0.9, box=(0, 0, 10, 10))]
    assert average_precision(preds, [], "cat", iou_threshold=0.5) == 0.0


def test_average_precision_false_positive_reduces_ap() -> None:
    gts = [GroundTruth(image_id="img0", class_name="cat", box=(0, 0, 10, 10))]
    preds = [
        # Higher-scoring false positive (no matching GT nearby) ranked first,
        # then the true positive -- should hurt AP relative to a clean TP-only case.
        Prediction(image_id="img0", class_name="cat", score=0.95, box=(50, 50, 60, 60)),
        Prediction(image_id="img0", class_name="cat", score=0.8, box=(0, 0, 10, 10)),
    ]
    ap = average_precision(preds, gts, "cat", iou_threshold=0.5)
    assert 0.0 < ap < 1.0


def test_average_precision_below_iou_threshold_is_false_positive() -> None:
    # Predicted box only barely overlaps the ground truth -- below 0.5 IoU.
    gts = [GroundTruth(image_id="img0", class_name="cat", box=(0, 0, 10, 10))]
    preds = [Prediction(image_id="img0", class_name="cat", score=0.9, box=(8, 8, 18, 18))]
    assert average_precision(preds, gts, "cat", iou_threshold=0.5) == 0.0


def test_average_precision_redundant_fp_after_full_recall_does_not_hurt_ap() -> None:
    # Two predictions for the same single ground truth -- only the first
    # (higher-scoring) can match; the second is a false positive, but it
    # comes *after* recall has already reached 1.0 from the first (correct)
    # detection. Under the standard precision-envelope AP definition this is
    # a real, well-known property: AP only cares about the best precision
    # achievable at each recall level, so a redundant FP ranked after full
    # recall doesn't lower it -- unlike a plain precision@threshold would.
    gts = [GroundTruth(image_id="img0", class_name="cat", box=(0, 0, 10, 10))]
    preds = [
        Prediction(image_id="img0", class_name="cat", score=0.9, box=(0, 0, 10, 10)),
        Prediction(image_id="img0", class_name="cat", score=0.8, box=(0, 0, 10, 10)),
    ]
    ap = average_precision(preds, gts, "cat", iou_threshold=0.5)
    assert ap == pytest.approx(1.0)


def test_mean_average_precision_averages_across_classes() -> None:
    gts = [
        GroundTruth(image_id="img0", class_name="cat", box=(0, 0, 10, 10)),
        GroundTruth(image_id="img0", class_name="dog", box=(20, 20, 30, 30)),
    ]
    preds = [
        Prediction(image_id="img0", class_name="cat", score=0.9, box=(0, 0, 10, 10)),
        # no prediction for "dog" -> AP(dog) = 0
    ]
    result = mean_average_precision(preds, gts, iou_threshold=0.5)
    assert result["per_class"]["cat"] == pytest.approx(1.0)
    assert result["per_class"]["dog"] == 0.0
    assert result["mAP"] == pytest.approx(0.5)


def test_mean_average_precision_empty_ground_truth_is_zero() -> None:
    result = mean_average_precision([], [], iou_threshold=0.5)
    assert result == {"mAP": 0.0, "per_class": {}}
