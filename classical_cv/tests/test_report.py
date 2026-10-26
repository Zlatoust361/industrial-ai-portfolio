"""Unit tests for `classical_cv.report`."""

from __future__ import annotations

from classical_cv.benchmark import evaluate_classical_cv
from classical_cv.defect_images import build_demo_dataset
from classical_cv.report import (
    CLIP_STATUS_BLOCKED,
    CLIP_STATUS_OK,
    ClipBaselineSummary,
    build_report,
    to_dict,
)
from src.synthetic.config import load_defects_config

DEFECTS_CONFIG = load_defects_config("config/defects.yaml")


def _classical_report():
    dataset = build_demo_dataset(DEFECTS_CONFIG, num_positive=10, num_negative=10, seed=42)
    return evaluate_classical_cv(dataset)


def test_build_report_with_blocked_clip_mentions_block_reason() -> None:
    classical = _classical_report()
    clip = ClipBaselineSummary(status=CLIP_STATUS_BLOCKED, blocked_reason="network unavailable")
    report = build_report(classical, clip)
    assert "network unavailable" in report.recommendation
    assert "blocked" not in report.recommendation.lower() or "blocked" in str(clip.status)


def test_build_report_with_measured_clip_compares_speed() -> None:
    classical = _classical_report()
    clip = ClipBaselineSummary(status=CLIP_STATUS_OK, accuracy=0.9, mean_latency_ms=120.0)
    report = build_report(classical, clip)
    assert "accuracy=0.90" in report.recommendation
    assert "faster" in report.recommendation


def test_to_dict_is_json_serializable() -> None:
    import json

    classical = _classical_report()
    clip = ClipBaselineSummary(status=CLIP_STATUS_BLOCKED, blocked_reason="offline")
    report = build_report(classical, clip)

    serialized = json.dumps(to_dict(report))
    assert "classical_cv" in serialized
    assert "clip_baseline" in serialized
    assert "recommendation" in serialized
