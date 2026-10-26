"""Unit tests for `modern_detectors.report`."""

from __future__ import annotations

import json

from modern_detectors.benchmark import DetectorBenchReport
from modern_detectors.owlvit_baseline import OwlVitSummary
from modern_detectors.report import build_report, to_dict


def _bench(detector_name: str, ap: float, latency_ms: float) -> DetectorBenchReport:
    return DetectorBenchReport(
        detector_name=detector_name,
        num_images=15,
        average_precision=ap,
        operating_point={"precision": ap, "recall": ap, "f1": ap, "true_positives": 1,
                          "false_positives": 1, "false_negatives": 1},
        mean_latency_ms=latency_ms,
        iou_threshold=0.5,
        score_threshold=0.3,
    )


def test_build_report_with_blocked_owlvit_mentions_block_reason() -> None:
    rt_detr = _bench("rt_detr_tiny", 0.2, 20.0)
    yolo = _bench("yolov8n_scratch", 0.1, 15.0)
    owlvit = OwlVitSummary(status="blocked", blocked_reason="network unavailable")

    report = build_report(rt_detr, yolo, owlvit)
    assert "network unavailable" in report.recommendation


def test_build_report_with_measured_owlvit_mentions_zero_training() -> None:
    rt_detr = _bench("rt_detr_tiny", 0.2, 20.0)
    yolo = _bench("yolov8n_scratch", 0.1, 15.0)
    owlvit = OwlVitSummary(status="measured", average_precision=0.3, mean_latency_ms=200.0)

    report = build_report(rt_detr, yolo, owlvit)
    assert "ZERO training images" in report.recommendation
    assert "0.30" in report.recommendation


def test_build_report_names_the_faster_and_higher_ap_detector() -> None:
    rt_detr = _bench("rt_detr_tiny", 0.3, 10.0)
    yolo = _bench("yolov8n_scratch", 0.1, 50.0)
    owlvit = OwlVitSummary(status="blocked", blocked_reason="offline")

    report = build_report(rt_detr, yolo, owlvit)
    assert "RT-DETR-tiny scored higher AP" in report.recommendation
    assert "RT-DETR-tiny ran faster" in report.recommendation


def test_to_dict_is_json_serializable() -> None:
    rt_detr = _bench("rt_detr_tiny", 0.2, 20.0)
    yolo = _bench("yolov8n_scratch", 0.1, 15.0)
    owlvit = OwlVitSummary(status="blocked", blocked_reason="offline")

    report = build_report(rt_detr, yolo, owlvit)
    serialized = json.dumps(to_dict(report))
    assert "rt_detr" in serialized
    assert "yolo" in serialized
    assert "owlvit" in serialized
