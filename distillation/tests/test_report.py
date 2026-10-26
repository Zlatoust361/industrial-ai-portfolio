"""Unit tests for `distillation.report`."""

from __future__ import annotations

import json

from distillation.benchmark import CompressionRow
from distillation.report import build_report, to_dict


def _row(technique: str, accuracy: float, params: int, size_kb: float = 10.0) -> CompressionRow:
    return CompressionRow(
        technique=technique, num_parameters=params, size_kb=size_kb, accuracy=accuracy,
        latency_ms=1.0,
    )


def test_build_report_notes_when_distillation_helps() -> None:
    rows = [
        _row("teacher", 1.0, 1000),
        _row("student_plain", 0.6, 100),
        _row("student_distilled", 0.9, 100),
    ]
    report = build_report(rows)
    assert "distillation helped" in report.recommendation


def test_build_report_notes_when_distillation_does_not_help() -> None:
    rows = [
        _row("teacher", 1.0, 1000),
        _row("student_plain", 0.9, 100),
        _row("student_distilled", 0.9, 100),
    ]
    report = build_report(rows)
    assert "did NOT measurably help" in report.recommendation


def test_build_report_notes_pruning_ratio() -> None:
    rows = [
        _row("teacher", 1.0, 1000),
        _row("pruned_50pct", 1.0, 250),
    ]
    report = build_report(rows)
    assert "4.0x fewer parameters" in report.recommendation


def test_build_report_flags_a_nas_search_that_underperforms_the_hand_picked_student() -> None:
    rows = [
        _row("student_plain", 1.0, 1265),
        _row("nas_best", 0.667, 45),
    ]
    report = build_report(rows)
    assert "scored WORSE than the hand-picked student" in report.recommendation


def test_build_report_handles_missing_rows_gracefully() -> None:
    report = build_report([_row("teacher", 1.0, 1000)])
    assert report.recommendation  # does not crash, says SOMETHING


def test_to_dict_is_json_serializable() -> None:
    rows = [_row("teacher", 1.0, 1000), _row("student_plain", 0.6, 100)]
    report = build_report(rows)
    serialized = json.dumps(to_dict(report))
    assert "teacher" in serialized
    assert "recommendation" in serialized
