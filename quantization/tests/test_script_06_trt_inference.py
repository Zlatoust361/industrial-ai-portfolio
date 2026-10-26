"""Script-level tests for `scripts/06_trt_inference.py`.

Same shape as `04`/`05`'s script-level tests: the Jetson gate itself, and
`--force` reaching exactly `pycuda`'s import boundary (the first thing
`utils.trt_inference.TrtImageEncoder.__init__` imports, ahead of
`tensorrt` -- see that module's docstring) once a real test-set directory
exists for `discover_test_images` to find.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from conftest import load_script_module, write_config
from PIL import Image

trt_inference_script = load_script_module("06_trt_inference.py")

SAMPLE_NV_TEGRA_RELEASE = "# R36 (release), REVISION: 4.0\n"


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
            "06_trt_inference.py",
            "--config",
            str(config_path),
            "--engine",
            "outputs/engine/image_encoder_fp16.engine",
            "--label",
            "fp16",
            "--marker-path",
            str(tmp_path / "no_such_marker"),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        trt_inference_script.main()

    assert exc_info.value.code == 1
    assert "06_trt_inference.py" in capsys.readouterr().err


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
            "06_trt_inference.py",
            "--config",
            str(config_path),
            "--engine",
            "outputs/engine/image_encoder_fp16.engine",
            "--label",
            "fp16",
            "--marker-path",
            str(tmp_path / "no_such_marker"),
            "--force",
        ],
    )

    with pytest.raises(ModuleNotFoundError, match="pycuda"):
        trt_inference_script.main()


def test_run_inference_benchmark_raises_clear_error_without_test_set(tmp_path: Path) -> None:
    from utils.config import load_config

    config = load_config(write_config(tmp_path))

    with pytest.raises(FileNotFoundError, match="dataset_pipeline"):
        trt_inference_script.run_inference_benchmark(
            config, tmp_path / "some.engine", None, None
        )
