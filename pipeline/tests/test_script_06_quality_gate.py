"""Script-level tests for scripts/06_quality_gate.py.

tests/test_quality_gate.py already covers `evaluate_quality_gate`'s pure
threshold logic; this covers the actual script -- reading
`outputs/interpretations.json`, writing `outputs/metrics/quality_gate.json`,
and (this is the part `run_all.sh`/CI actually depend on) exiting `0` for
GREEN/YELLOW and `1` for RED.

Run with (from pipeline/):
    PYTHONPATH=. pytest tests/test_script_06_quality_gate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from conftest import load_script_module, write_pipeline_config

quality_gate = load_script_module("06_quality_gate.py")


def _write_interpretations(tmp_path: Path, defect_rate: float) -> None:
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir(parents=True)
    (outputs_dir / "interpretations.json").write_text(
        json.dumps(
            {
                "summary": {"total": 10, "defect_count": 0, "defect_rate": defect_rate},
                "verdicts": [],
            }
        )
    )


@pytest.mark.parametrize(
    ("defect_rate", "expected_status", "expected_exit"),
    [
        (0.0, "GREEN", 0),
        (0.05, "YELLOW", 0),
        (0.5, "RED", 1),
    ],
)
def test_main_exit_code_matches_status(
    tmp_path: Path, monkeypatch, defect_rate: float, expected_status: str, expected_exit: int
) -> None:
    config_path = write_pipeline_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    _write_interpretations(tmp_path, defect_rate)
    monkeypatch.setattr(sys, "argv", ["06_quality_gate.py", "--config", str(config_path)])

    if expected_exit == 0:
        quality_gate.main()  # must NOT raise SystemExit
    else:
        with pytest.raises(SystemExit) as exc_info:
            quality_gate.main()
        assert exc_info.value.code == expected_exit

    metrics = json.loads((tmp_path / "outputs" / "metrics" / "quality_gate.json").read_text())
    assert metrics["status"] == expected_status
    assert metrics["defect_rate"] == pytest.approx(defect_rate)


def test_main_missing_interpretations_exits_with_error(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    config_path = write_pipeline_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["06_quality_gate.py", "--config", str(config_path)])

    with pytest.raises(SystemExit) as exc_info:
        quality_gate.main()

    assert exc_info.value.code == 1
    assert "05_interpret.py" in capsys.readouterr().err
