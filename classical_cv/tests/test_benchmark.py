"""Unit tests for `classical_cv.benchmark`."""

from __future__ import annotations

from classical_cv.benchmark import (
    _percentile,
    evaluate_classical_cv,
    run_detector_on_dataset,
)
from classical_cv.defect_images import build_demo_dataset
from src.synthetic.config import load_defects_config

DEFECTS_CONFIG = load_defects_config("config/defects.yaml")


def test_percentile_nearest_rank() -> None:
    assert _percentile([], 0.95) == 0.0
    assert _percentile([1.0, 2.0, 3.0, 4.0], 0.0) == 1.0
    assert _percentile([1.0, 2.0, 3.0, 4.0], 1.0) == 4.0


def test_run_detector_on_dataset_measures_nonnegative_latency() -> None:
    dataset = build_demo_dataset(DEFECTS_CONFIG, num_positive=3, num_negative=3, seed=42)
    results = run_detector_on_dataset(dataset)
    assert len(results) == len(dataset)
    for result, image in zip(results, dataset, strict=True):
        assert result.image_id == image.image_id
        assert result.latency_ms >= 0.0


def test_evaluate_classical_cv_reports_real_measured_numbers() -> None:
    """Real thresholds from this detector's own measured behaviour at
    seed=42 (see `classical_cv/README.md`) -- a sanity floor, not a claim
    of a specific target accuracy."""
    dataset = build_demo_dataset(DEFECTS_CONFIG, num_positive=30, num_negative=30, seed=42)
    report = evaluate_classical_cv(dataset)

    assert report.num_images == 60
    assert report.num_positive == 30
    assert 0.0 < report.average_precision <= 1.0
    assert report.operating_point["precision"] > 0.0
    assert report.operating_point["recall"] > 0.0
    assert report.mean_latency_ms > 0.0
    assert report.p95_latency_ms >= report.mean_latency_ms
    assert len(report.per_image) == 60


def test_evaluate_classical_cv_is_deterministic_for_the_same_seed() -> None:
    dataset = build_demo_dataset(DEFECTS_CONFIG, num_positive=10, num_negative=10, seed=7)
    first = evaluate_classical_cv(dataset)
    second = evaluate_classical_cv(dataset)
    assert first.average_precision == second.average_precision
    assert first.operating_point == second.operating_point
