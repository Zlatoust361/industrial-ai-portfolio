"""Tests for `scripts/run_modern_detectors_demo.py` -- the week 26 one-command deliverable.

Uses small `num_positive`/`num_negative`/`rt_detr_steps`/`yolo_epochs`
overrides throughout -- `make demo`'s real, illustrative-scale numbers
(200 RT-DETR steps, 60 YOLO epochs) are documented in
`modern_detectors/README.md`, not re-run here (would take minutes).
"""

from __future__ import annotations

import json
from pathlib import Path

from conftest import load_script_module

run_modern_detectors_demo = load_script_module("run_modern_detectors_demo.py")


def test_run_demo_with_skip_owlvit_reports_blocked_by_flag(tmp_path: Path) -> None:
    report = run_modern_detectors_demo.run_modern_detectors_demo(
        num_positive=6,
        num_negative=4,
        val_fraction=0.3,
        rt_detr_steps=3,
        yolo_epochs=1,
        seed=42,
        defects_config_path="config/defects.yaml",
        dataset_dir=str(tmp_path / "yolo_ds"),
        skip_owlvit=True,
    )

    assert report["owlvit"]["status"] == "blocked"
    assert "--skip-owlvit" in report["owlvit"]["blocked_reason"]
    assert "rt_detr" in report
    assert "yolo" in report
    assert "recommendation" in report
    assert report["num_train_images"] + report["num_val_images"] == 10


def test_run_demo_report_has_real_numbers(tmp_path: Path) -> None:
    report = run_modern_detectors_demo.run_modern_detectors_demo(
        num_positive=6,
        num_negative=4,
        val_fraction=0.3,
        rt_detr_steps=3,
        yolo_epochs=1,
        seed=42,
        defects_config_path="config/defects.yaml",
        dataset_dir=str(tmp_path / "yolo_ds"),
        skip_owlvit=True,
    )

    assert 0.0 <= report["rt_detr"]["average_precision"] <= 1.0
    assert 0.0 <= report["yolo"]["average_precision"] <= 1.0
    assert report["rt_detr"]["mean_latency_ms"] > 0.0
    assert report["yolo"]["mean_latency_ms"] > 0.0
    assert len(report["rt_detr"]["train_losses"]) == 3
    assert report["yolo"]["train_seconds"] > 0.0


def test_main_writes_report(tmp_path: Path, monkeypatch) -> None:
    output_path = tmp_path / "report.json"
    monkeypatch.setattr(
        run_modern_detectors_demo,
        "parse_args",
        lambda: run_modern_detectors_demo.argparse.Namespace(
            num_positive=6,
            num_negative=4,
            val_fraction=0.3,
            rt_detr_steps=3,
            yolo_epochs=1,
            seed=42,
            defects_config="config/defects.yaml",
            dataset_dir=str(tmp_path / "yolo_ds"),
            skip_owlvit=True,
            output=str(output_path),
        ),
    )

    run_modern_detectors_demo.main()

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["owlvit"]["status"] == "blocked"
    assert "rt_detr" in report
