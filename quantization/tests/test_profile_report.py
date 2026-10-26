"""Unit tests for `utils/profile_report.py` -- fully real, no tensorrt/pycuda/torch involved."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from utils.profile_report import (
    SpeedupRow,
    aggregate_peak_temperatures,
    compute_speedup_rows,
    discover_trt_inference_reports,
    group_by_power_mode,
    is_thermal_warning,
    load_optional_json,
    power_mode_name,
    render_summary_markdown,
)


def _report(
    label: str,
    mode_name: str | None,
    mean_ms: float,
    p95_ms: float,
    throughput_ips: float,
    max_temps: dict[str, float] | None = None,
    mean_temps: dict[str, float] | None = None,
) -> dict:
    return {
        "metadata": {
            "label": label,
            "power_mode": {"mode_name": mode_name, "mode_id": 1} if mode_name else None,
        },
        "latency": {
            "batch_size": 1,
            "mean_ms": mean_ms,
            "p50_ms": mean_ms,
            "p95_ms": p95_ms,
            "throughput_ips": throughput_ips,
        },
        "temperatures": {
            "mean_temps_c": mean_temps or {},
            "max_temps_c": max_temps or {},
        },
    }


def _write_reports(inference_dir: Path, reports: list[dict]) -> None:
    inference_dir.mkdir(parents=True, exist_ok=True)
    for index, report in enumerate(reports):
        (inference_dir / f"report_{index:02d}.json").write_text(json.dumps(report))


def test_discover_trt_inference_reports_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="06_trt_inference"):
        discover_trt_inference_reports(tmp_path / "does_not_exist")


def test_discover_trt_inference_reports_empty_dir_raises(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()

    with pytest.raises(FileNotFoundError, match="06_trt_inference"):
        discover_trt_inference_reports(tmp_path / "empty")


def test_discover_trt_inference_reports_loads_real_json(tmp_path: Path) -> None:
    reports = [_report("fp16", "15W", 10.0, 12.0, 100.0), _report("int4", "15W", 5.0, 6.0, 200.0)]
    _write_reports(tmp_path, reports)

    loaded = discover_trt_inference_reports(tmp_path)

    assert len(loaded) == 2
    assert {r["metadata"]["label"] for r in loaded} == {"fp16", "int4"}


def test_load_optional_json_missing_returns_none(tmp_path: Path) -> None:
    assert load_optional_json(tmp_path / "nope.json") is None


def test_load_optional_json_existing_loads(tmp_path: Path) -> None:
    path = tmp_path / "x.json"
    path.write_text(json.dumps({"a": 1}))

    assert load_optional_json(path) == {"a": 1}


def test_power_mode_name_present() -> None:
    assert power_mode_name(_report("fp16", "MAXN_SUPER", 1, 1, 1)) == "MAXN_SUPER"


def test_power_mode_name_missing_defaults_to_unknown() -> None:
    assert power_mode_name(_report("fp16", None, 1, 1, 1)) == "unknown"


def test_group_by_power_mode() -> None:
    reports = [
        _report("fp16", "15W", 10, 12, 100),
        _report("int4", "15W", 5, 6, 200),
        _report("fp16", "MAXN_SUPER", 8, 9, 120),
    ]

    grouped = group_by_power_mode(reports)

    assert set(grouped) == {"15W", "MAXN_SUPER"}
    assert set(grouped["15W"]) == {"fp16", "int4"}
    assert set(grouped["MAXN_SUPER"]) == {"fp16"}


def test_compute_speedup_rows_complete_pair() -> None:
    grouped = group_by_power_mode(
        [_report("fp16", "15W", 10.0, 12.0, 100.0), _report("int4", "15W", 5.0, 6.0, 200.0)]
    )

    rows = compute_speedup_rows(grouped)

    assert len(rows) == 1
    row = rows[0]
    assert row.power_mode == "15W"
    assert row.note is None
    assert row.speedup_mean == pytest.approx(2.0)
    assert row.speedup_p95 == pytest.approx(2.0)
    assert row.fp16_mean_ms == 10.0
    assert row.int4_mean_ms == 5.0


def test_compute_speedup_rows_missing_side_gets_note() -> None:
    grouped = group_by_power_mode([_report("fp16", "15W", 10.0, 12.0, 100.0)])

    rows = compute_speedup_rows(grouped)

    assert len(rows) == 1
    assert rows[0].note is not None
    assert "int4" in rows[0].note
    assert rows[0].speedup_mean is None


def test_compute_speedup_rows_zero_int4_latency_is_safe() -> None:
    grouped = group_by_power_mode(
        [_report("fp16", "15W", 10.0, 12.0, 100.0), _report("int4", "15W", 0.0, 0.0, 0.0)]
    )

    rows = compute_speedup_rows(grouped)

    assert rows[0].speedup_mean is None
    assert rows[0].speedup_p95 is None


def test_compute_speedup_rows_sorted_by_mode() -> None:
    grouped = group_by_power_mode(
        [
            _report("fp16", "MAXN_SUPER", 8, 9, 1),
            _report("int4", "MAXN_SUPER", 4, 4.5, 1),
            _report("fp16", "15W", 10, 12, 1),
            _report("int4", "15W", 5, 6, 1),
        ]
    )

    rows = compute_speedup_rows(grouped)

    assert [row.power_mode for row in rows] == ["15W", "MAXN_SUPER"]


def test_aggregate_peak_temperatures_empty_when_no_readings() -> None:
    reports = [_report("fp16", "15W", 10, 12, 1)]

    assert aggregate_peak_temperatures(reports) == {}


def test_aggregate_peak_temperatures_takes_max_across_reports() -> None:
    reports = [
        _report("fp16", "15W", 10, 12, 1, max_temps={"GPU": 40.0, "CPU": 45.0}),
        _report("int4", "15W", 5, 6, 1, max_temps={"GPU": 55.0, "CPU": 42.0}),
    ]

    peaks = aggregate_peak_temperatures(reports)

    assert peaks == {"GPU": 55.0, "CPU": 45.0}


def test_is_thermal_warning_below_threshold() -> None:
    assert is_thermal_warning({"GPU": 40.0}, threshold_c=85.0) is False


def test_is_thermal_warning_at_or_above_threshold() -> None:
    assert is_thermal_warning({"GPU": 85.0}, threshold_c=85.0) is True
    assert is_thermal_warning({"GPU": 90.0}, threshold_c=85.0) is True


def test_render_summary_markdown_includes_key_sections() -> None:
    reports = [
        _report(
            "fp16", "15W", 10.0, 12.0, 100.0, max_temps={"GPU": 50.0}, mean_temps={"GPU": 45.0}
        ),
        _report(
            "int4", "15W", 5.0, 6.0, 200.0, max_temps={"GPU": 52.0}, mean_temps={"GPU": 46.0}
        ),
    ]
    grouped = group_by_power_mode(reports)
    speedup_rows = compute_speedup_rows(grouped)
    peaks = aggregate_peak_temperatures(reports)

    markdown = render_summary_markdown(
        reports=reports,
        grouped=grouped,
        speedup_rows=speedup_rows,
        quality_report={
            "mean_cosine_similarity": 0.995,
            "passed": True,
            "num_images": 64,
        },
        baseline_report=None,
        peak_temps_c=peaks,
        throttle_temp_c=85.0,
        generated_at="2026-08-22T00:00:00+00:00",
    )

    assert "# Week 09" in markdown
    assert "## Speedup: FP16 → INT4" in markdown
    assert "2.00x" in markdown  # speedup_mean
    assert "## Trade-off: latency vs quality" in markdown
    assert "0.9950" in markdown  # quality similarity
    assert "PASSED" in markdown
    assert "## Temperature / throttling" in markdown
    assert "52.0" in markdown  # peak GPU temp
    assert "Thermal warning triggered:** no" in markdown
    assert "### Per-run breakdown" in markdown
    assert "No `01_baseline_inference.py` report found" in markdown


def test_render_summary_markdown_handles_missing_quality_and_temps() -> None:
    reports = [_report("fp16", "15W", 10.0, 12.0, 100.0), _report("int4", "15W", 5.0, 6.0, 200.0)]
    grouped = group_by_power_mode(reports)
    speedup_rows = compute_speedup_rows(grouped)

    markdown = render_summary_markdown(
        reports=reports,
        grouped=grouped,
        speedup_rows=speedup_rows,
        quality_report=None,
        baseline_report=None,
        peak_temps_c={},
        throttle_temp_c=85.0,
        generated_at="2026-08-22T00:00:00+00:00",
    )

    assert "No `07_evaluate_quality.py` report found" in markdown
    assert "No `tegrastats` readings in any report" in markdown


def test_render_summary_markdown_thermal_warning_triggers() -> None:
    reports = [_report("fp16", "15W", 10.0, 12.0, 100.0, max_temps={"GPU": 90.0})]
    grouped = group_by_power_mode(reports)
    speedup_rows = compute_speedup_rows(grouped)
    peaks = aggregate_peak_temperatures(reports)

    markdown = render_summary_markdown(
        reports=reports,
        grouped=grouped,
        speedup_rows=speedup_rows,
        quality_report=None,
        baseline_report=None,
        peak_temps_c=peaks,
        throttle_temp_c=85.0,
        generated_at="2026-08-22T00:00:00+00:00",
    )

    assert "Thermal warning triggered:** YES" in markdown
    assert "⚠️" in markdown


def test_render_summary_markdown_includes_baseline() -> None:
    reports = [_report("fp16", "15W", 10.0, 12.0, 100.0)]
    grouped = group_by_power_mode(reports)
    speedup_rows = compute_speedup_rows(grouped)

    markdown = render_summary_markdown(
        reports=reports,
        grouped=grouped,
        speedup_rows=speedup_rows,
        quality_report=None,
        baseline_report={
            "metadata": {"precision": "fp16"},
            "results": [{"batch_size": 1, "mean_ms": 15.5}],
        },
        peak_temps_c={},
        throttle_temp_c=85.0,
        generated_at="2026-08-22T00:00:00+00:00",
    )

    assert "15.50 ms" in markdown


def test_speedup_row_is_frozen_dataclass() -> None:
    row = SpeedupRow("15W", 1, 1, 1, 1, 1, 1, 1, 1, None)
    with pytest.raises(AttributeError):
        row.power_mode = "MAXN_SUPER"  # type: ignore[misc]
