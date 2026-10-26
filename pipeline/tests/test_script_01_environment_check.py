"""Script-level tests for scripts/01_environment_check.py.

The core ask this file covers: a mocked "we are on a Jetson Orin Nano" run
and a mocked "we are not" run, proving the script's actual CLI/main()
wiring (not just utils.environment's pure functions, already covered by
tests/test_environment.py) reaches the right branch end to end --
`build_environment_report()`'s `marker_path` default is bound once at
import time (a real Python gotcha), so the script exposes `--marker-path`
specifically so tests can override it without monkeypatching internals.

Run with (from pipeline/):
    PYTHONPATH=. pytest tests/test_script_01_environment_check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from conftest import load_script_module, write_pipeline_config

environment_check = load_script_module("01_environment_check.py")

SAMPLE_NV_TEGRA_RELEASE = (
    "# R36 (release), REVISION: 4.0, GCID: 38968081, BOARD: generic, "
    "EABI: aarch64, DATE: Fri Jun 14 00:00:00 UTC 2024\n"
)


def _run(monkeypatch, tmp_path: Path, marker_path: Path) -> dict[str, object]:
    config_path = write_pipeline_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "01_environment_check.py",
            "--config",
            str(config_path),
            "--marker-path",
            str(marker_path),
        ],
    )
    environment_check.main()
    return json.loads((tmp_path / "outputs" / "env_report.json").read_text())


def test_main_off_jetson_when_marker_missing(tmp_path: Path, monkeypatch) -> None:
    report = _run(monkeypatch, tmp_path, tmp_path / "no_such_marker")

    assert report["is_jetson"] is False
    assert report["l4t_version"] is None
    assert report["jetpack_generation"] is None
    assert report["power_mode"] is None


def test_main_on_jetson_when_marker_present(tmp_path: Path, monkeypatch) -> None:
    marker = tmp_path / "nv_tegra_release"
    marker.write_text(SAMPLE_NV_TEGRA_RELEASE)

    report = _run(monkeypatch, tmp_path, marker)

    assert report["is_jetson"] is True
    assert report["l4t_version"] == "R36.4.0"
    assert report["jetpack_generation"] == "6.x"


def test_main_prints_off_device_warning_only_when_not_jetson(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _run(monkeypatch, tmp_path, tmp_path / "no_such_marker")
    assert "Not running on a Jetson" in capsys.readouterr().err


def test_main_stays_quiet_about_off_device_when_on_jetson(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    marker = tmp_path / "nv_tegra_release"
    marker.write_text(SAMPLE_NV_TEGRA_RELEASE)

    _run(monkeypatch, tmp_path, marker)

    assert "Not running on a Jetson" not in capsys.readouterr().err
