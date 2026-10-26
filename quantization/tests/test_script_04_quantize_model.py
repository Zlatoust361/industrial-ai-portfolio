"""Script-level tests for `scripts/04_quantize_model.py`.

Two things verified for real: the Jetson gate itself (mocked "on a Jetson"
vs "not on a Jetson", via `--marker-path`, same technique as
`pipeline/tests/test_script_01_environment_check.py`), and that `--force`
on a non-Jetson machine reaches exactly `tensorrt`'s import (not
pip-installable in this sandbox -- see `../README.md`) rather than some
earlier, unrelated failure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from conftest import load_script_module, write_config

quantize_model = load_script_module("04_quantize_model.py")

SAMPLE_NV_TEGRA_RELEASE = "# R36 (release), REVISION: 4.0\n"


def _write_calibration_manifest(outputs_dir: Path) -> None:
    calibration_dir = outputs_dir / "calibration_data"
    calibration_dir.mkdir(parents=True, exist_ok=True)
    (calibration_dir / "manifest.json").write_text(
        json.dumps({"images": [{"output": str(calibration_dir / "calib_0000.png")}]})
    )


def test_main_exits_when_not_jetson_and_not_forced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = write_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "04_quantize_model.py",
            "--config",
            str(config_path),
            "--marker-path",
            str(tmp_path / "no_such_marker"),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        quantize_model.main()

    assert exc_info.value.code == 1
    assert "Jetson" in capsys.readouterr().err


def test_main_proceeds_when_marker_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    marker = tmp_path / "nv_tegra_release"
    marker.write_text(SAMPLE_NV_TEGRA_RELEASE)
    config_path = write_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "04_quantize_model.py",
            "--config",
            str(config_path),
            "--marker-path",
            str(marker),
        ],
    )
    _write_calibration_manifest(tmp_path / "outputs")

    with pytest.raises(ModuleNotFoundError, match="tensorrt"):
        quantize_model.main()


def test_main_force_on_dev_machine_reaches_tensorrt_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = write_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "04_quantize_model.py",
            "--config",
            str(config_path),
            "--marker-path",
            str(tmp_path / "no_such_marker"),
            "--force",
        ],
    )
    _write_calibration_manifest(tmp_path / "outputs")

    with pytest.raises(ModuleNotFoundError, match="tensorrt"):
        quantize_model.main()


def test_run_calibration_raises_clear_error_without_calibration_manifest(tmp_path: Path) -> None:
    from utils.config import load_config

    config = load_config(write_config(tmp_path))

    with pytest.raises(FileNotFoundError, match="03_prepare_calibration_data"):
        quantize_model.run_calibration(config, None)
