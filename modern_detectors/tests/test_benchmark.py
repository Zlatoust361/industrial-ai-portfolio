"""Unit tests for `modern_detectors.benchmark`."""

from __future__ import annotations

from classical_cv.defect_images import build_demo_dataset
from modern_detectors.benchmark import measure_per_image_latency_ms, score_predictions
from modern_detectors.dataset import resize_dataset
from src.evaluation.detection_metrics import Prediction
from src.synthetic.config import load_defects_config

DEFECTS_CONFIG = load_defects_config("config/defects.yaml")


def _tiny_dataset():
    dataset = build_demo_dataset(DEFECTS_CONFIG, num_positive=3, num_negative=2, seed=42)
    return resize_dataset(dataset, (512, 512))


def test_score_predictions_perfect_detector_gets_ap_one() -> None:
    images = _tiny_dataset()
    predictions_by_image = {
        image.image_id: [
            Prediction(image_id=image.image_id, class_name="defect", score=0.9, box=box)
            for box in image.ground_truth_boxes
        ]
        for image in images
    }
    report = score_predictions(
        "perfect", predictions_by_image, images, latencies_ms=[1.0] * len(images)
    )
    assert report.average_precision == 1.0
    assert report.num_images == len(images)


def test_score_predictions_empty_predictions_get_ap_zero() -> None:
    images = _tiny_dataset()
    report = score_predictions(
        "empty", {image.image_id: [] for image in images}, images, latencies_ms=[1.0] * len(images)
    )
    assert report.average_precision == 0.0
    assert report.operating_point["true_positives"] == 0


def test_measure_per_image_latency_ms_returns_one_value_per_image() -> None:
    images = _tiny_dataset()
    latencies = measure_per_image_latency_ms(lambda image: len(image.ground_truth_boxes), images)
    assert len(latencies) == len(images)
    assert all(latency >= 0.0 for latency in latencies)
