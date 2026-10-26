"""Script-level tests for scripts/07_generate_report.py.

tests/test_report.py already covers `render_report`'s pure templating; this
covers the actual script -- reading all three upstream artifacts
(`env_report.json`, `interpretations.json`, `metrics/quality_gate.json`)
and writing `outputs/report.md`, including each "run the earlier stage
first" error path.

Run with (from pipeline/):
    PYTHONPATH=. pytest tests/test_script_07_generate_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from conftest import load_script_module, write_pipeline_config

generate_report = load_script_module("07_generate_report.py")


def _write_upstream_artifacts(tmp_path: Path) -> None:
    outputs_dir = tmp_path / "outputs"
    (outputs_dir / "metrics").mkdir(parents=True)
    (outputs_dir / "env_report.json").write_text(
        json.dumps(
            {
                "is_jetson": False,
                "l4t_version": None,
                "jetpack_generation": None,
                "available_memory_mb": 1000.0,
                "power_mode": None,
            }
        )
    )
    (outputs_dir / "interpretations.json").write_text(
        json.dumps(
            {
                "summary": {"total": 2, "defect_count": 1, "defect_rate": 0.5},
                "verdicts": [
                    {"image": "a.jpg", "defect_score": 0.9, "verdict": "defect"},
                    {"image": "b.jpg", "defect_score": 0.1, "verdict": "ok"},
                ],
            }
        )
    )
    (outputs_dir / "metrics" / "quality_gate.json").write_text(
        json.dumps(
            {
                "status": "RED",
                "defect_rate": 0.5,
                "thresholds": {"green_max_defect_rate": 0.02, "yellow_max_defect_rate": 0.10},
            }
        )
    )


def test_main_writes_report_from_upstream_artifacts(tmp_path: Path, monkeypatch) -> None:
    config_path = write_pipeline_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    _write_upstream_artifacts(tmp_path)
    monkeypatch.setattr(sys, "argv", ["07_generate_report.py", "--config", str(config_path)])

    generate_report.main()

    report = (tmp_path / "outputs" / "report.md").read_text()
    assert "**Status: RED**" in report
    assert "a.jpg" in report
    assert "b.jpg" not in report  # only flagged (defect) images get a row


@pytest.mark.parametrize("missing_file", ["env_report.json", "interpretations.json"])
def test_main_missing_upstream_artifact_exits_with_error(
    tmp_path: Path, monkeypatch, capsys, missing_file: str
) -> None:
    config_path = write_pipeline_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    _write_upstream_artifacts(tmp_path)
    (tmp_path / "outputs" / missing_file).unlink()
    monkeypatch.setattr(sys, "argv", ["07_generate_report.py", "--config", str(config_path)])

    with pytest.raises(SystemExit) as exc_info:
        generate_report.main()

    assert exc_info.value.code == 1
    assert missing_file in capsys.readouterr().err
