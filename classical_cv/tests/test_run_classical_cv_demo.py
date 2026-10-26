"""Tests for `scripts/run_classical_cv_demo.py` -- the week 25 one-command deliverable."""

from __future__ import annotations

import json
from pathlib import Path

from conftest import load_script_module

run_classical_cv_demo = load_script_module("run_classical_cv_demo.py")


def test_run_demo_with_skip_clip_reports_blocked_by_flag() -> None:
    report = run_classical_cv_demo.run_classical_cv_demo(
        num_positive=3,
        num_negative=3,
        seed=42,
        defects_config_path="config/defects.yaml",
        skip_clip=True,
    )

    assert report["clip_baseline"]["status"] == "blocked"
    assert "--skip-clip" in report["clip_baseline"]["blocked_reason"]
    assert report["classical_cv"]["num_images"] == 6
    assert "recommendation" in report


def test_run_demo_report_has_real_classical_cv_numbers() -> None:
    report = run_classical_cv_demo.run_classical_cv_demo(
        num_positive=5,
        num_negative=5,
        seed=42,
        defects_config_path="config/defects.yaml",
        skip_clip=True,
    )

    classical = report["classical_cv"]
    assert 0.0 <= classical["average_precision"] <= 1.0
    assert classical["mean_latency_ms"] > 0.0


def test_main_writes_report(tmp_path: Path, monkeypatch) -> None:
    output_path = tmp_path / "report.json"
    monkeypatch.setattr(
        run_classical_cv_demo,
        "parse_args",
        lambda: run_classical_cv_demo.argparse.Namespace(
            num_positive=3,
            num_negative=3,
            seed=42,
            defects_config="config/defects.yaml",
            skip_clip=True,
            output=str(output_path),
        ),
    )

    run_classical_cv_demo.main()

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["classical_cv"]["num_images"] == 6
    assert report["clip_baseline"]["status"] == "blocked"
