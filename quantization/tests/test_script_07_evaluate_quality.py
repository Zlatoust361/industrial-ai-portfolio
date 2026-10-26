"""Script-level tests for `scripts/07_evaluate_quality.py`.

Same shape as `06`'s script-level tests: the Jetson gate itself, and
`--force` reaching exactly `pycuda`'s import boundary once a real
test-set directory exists.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from conftest import load_script_module, write_config
from PIL import Image

evaluate_quality_script = load_script_module("07_evaluate_quality.py")


def _make_test_set(test_set_dir: Path, count: int = 4) -> None:
    test_set_dir.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        Image.new("RGB", (8, 8)).save(test_set_dir / f"img_{index:03d}.jpg")


def test_main_exits_when_not_jetson_and_not_forced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = write_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "07_evaluate_quality.py",
            "--config",
            str(config_path),
            "--marker-path",
            str(tmp_path / "no_such_marker"),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        evaluate_quality_script.main()

    assert exc_info.value.code == 1
    assert "07_evaluate_quality.py" in capsys.readouterr().err


def test_main_force_reaches_pycuda_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    test_set_dir = tmp_path / "test_set"
    _make_test_set(test_set_dir)
    config_path = write_config(tmp_path, evaluation_test_set_dir=str(test_set_dir))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "07_evaluate_quality.py",
            "--config",
            str(config_path),
            "--marker-path",
            str(tmp_path / "no_such_marker"),
            "--force",
        ],
    )

    with pytest.raises(ModuleNotFoundError, match="pycuda"):
        evaluate_quality_script.main()


def test_evaluate_quality_raises_clear_error_without_test_set(tmp_path: Path) -> None:
    from utils.config import load_config

    config = load_config(write_config(tmp_path))

    with pytest.raises(FileNotFoundError, match="dataset_pipeline"):
        evaluate_quality_script.evaluate_quality(config, None, None)
