"""Tests for `scripts/run_distillation_demo.py` -- the week 28 one-command deliverable.

Uses small overrides throughout -- `make demo`'s real, illustrative-scale
numbers are documented in `distillation/README.md`, not re-run here
(would take minutes: a teacher, two students, a NAS search over 8
candidates, and a quantization pass, all real training).
"""

from __future__ import annotations

import json
from pathlib import Path

from conftest import load_script_module

run_distillation_demo = load_script_module("run_distillation_demo.py")


def test_run_demo_returns_all_expected_rows(tmp_path: Path) -> None:
    report = run_distillation_demo.run_distillation_demo(
        num_positive=6,
        num_negative=4,
        val_fraction=0.3,
        teacher_steps=30,
        student_steps=30,
        nas_steps=10,
        defects_config_path="config/defects.yaml",
        seed=42,
    )

    techniques = {row["technique"] for row in report["rows"]}
    assert techniques == {
        "teacher",
        "student_plain",
        "student_distilled",
        "pruned_50pct",
        "nas_best",
        "int8_quantized",
    }
    assert "recommendation" in report
    assert "nas_best_widths" in report
    assert len(report["nas_search_space"]) == 8  # DEFAULT_SEARCH_SPACE size


def test_main_writes_report(tmp_path: Path, monkeypatch) -> None:
    output_path = tmp_path / "report.json"
    monkeypatch.setattr(
        run_distillation_demo,
        "parse_args",
        lambda: run_distillation_demo.argparse.Namespace(
            num_positive=6,
            num_negative=4,
            val_fraction=0.3,
            teacher_steps=30,
            student_steps=30,
            nas_steps=10,
            defects_config="config/defects.yaml",
            seed=42,
            output=str(output_path),
        ),
    )

    run_distillation_demo.main()

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(report["rows"]) == 6
