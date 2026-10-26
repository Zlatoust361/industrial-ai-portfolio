"""Script-level tests for `scripts/05_build_engine.py`.

Same shape as `test_script_04_quantize_model.py`: the Jetson gate itself,
plus `--force` reaching exactly `tensorrt`'s import boundary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from conftest import load_script_module, write_config

build_engine_script = load_script_module("05_build_engine.py")

SAMPLE_NV_TEGRA_RELEASE = "# R36 (release), REVISION: 4.0\n"


def _write_calibration_fixtures(outputs_dir: Path) -> None:
    calibration_dir = outputs_dir / "calibration_data"
    calibration_dir.mkdir(parents=True, exist_ok=True)
    (calibration_dir / "manifest.json").write_text(
        json.dumps({"images": [{"output": str(calibration_dir / "calib_0000.png")}]})
    )
    cache_path = outputs_dir / "calibration" / "int4_calibration.cache"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(b"fake-cache-bytes")


def test_main_exits_when_not_jetson_and_not_forced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = write_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "05_build_engine.py",
            "--config",
            str(config_path),
            "--marker-path",
            str(tmp_path / "no_such_marker"),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        build_engine_script.main()

    assert exc_info.value.code == 1
    assert "05_build_engine.py" in capsys.readouterr().err


def test_main_force_on_dev_machine_reaches_pycuda_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = write_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "05_build_engine.py",
            "--config",
            str(config_path),
            "--marker-path",
            str(tmp_path / "no_such_marker"),
            "--force",
        ],
    )
    _write_calibration_fixtures(tmp_path / "outputs")

    # build_calibrator() (needed for the default, non-fp16-only path) imports
    # pycuda before parse_onnx_network() ever imports tensorrt -- pycuda is
    # the actual first boundary hit here, not tensorrt (see
    # test_main_fp16_only_force_reaches_tensorrt_boundary_without_calibration_fixtures
    # below for the path that does reach tensorrt first).
    with pytest.raises(ModuleNotFoundError, match="pycuda"):
        build_engine_script.main()


def test_build_engine_raises_clear_error_without_calibration_cache(tmp_path: Path) -> None:
    from utils.config import load_config

    config = load_config(write_config(tmp_path))

    with pytest.raises(FileNotFoundError, match="04_quantize_model"):
        build_engine_script.build_engine(config, None, fp16_only=False)


def test_main_fp16_only_force_reaches_tensorrt_boundary_without_calibration_fixtures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = write_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "05_build_engine.py",
            "--config",
            str(config_path),
            "--marker-path",
            str(tmp_path / "no_such_marker"),
            "--force",
            "--fp16-only",
        ],
    )
    # Deliberately no calibration cache/manifest fixtures -- --fp16-only
    # must not need them at all.

    with pytest.raises(ModuleNotFoundError, match="tensorrt"):
        build_engine_script.main()


def test_build_engine_fp16_only_does_not_require_calibration_cache(tmp_path: Path) -> None:
    from utils.config import load_config

    config = load_config(write_config(tmp_path))

    # No calibration cache/manifest written -- fp16_only=True must skip
    # straight past the calibration-cache check to parse_onnx_network
    # (which imports tensorrt before even checking the ONNX path exists),
    # not raise the calibration-cache error test_build_engine_raises_
    # clear_error_without_calibration_cache above covers for fp16_only=False.
    with pytest.raises(ModuleNotFoundError, match="tensorrt"):
        build_engine_script.build_engine(config, None, fp16_only=True)


def test_build_engine_workspace_mb_override_reaches_configure_builder_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--workspace-mb must actually reach configure_builder_config, not just parse.

    Real finding on a Jetson Orin Nano: the orin_nano_8gb profile's default
    1024 MB workspace still hit CUDA out-of-memory at build time when the
    device's free unified memory was lower than usual -- workspace_mb_override
    exists so a user can lower it without editing config.yaml. Stubs out
    parse_onnx_network/configure_builder_config (no real tensorrt/pycuda in
    this sandbox) to capture the workspace_size_mb actually passed through,
    rather than just asserting no exception.
    """
    from utils.config import load_config

    config = load_config(write_config(tmp_path))
    assert config.get_profile(None).workspace_size_mb == 1024  # the override below must differ

    captured: dict[str, object] = {}

    def fake_parse_onnx_network(onnx_path: Path) -> tuple[object, object, object]:
        return object(), object(), object()

    def fake_configure_builder_config(builder: object, network: object, **kwargs: object) -> object:
        captured["workspace_size_mb"] = kwargs["workspace_size_mb"]
        raise RuntimeError("stop-before-real-build")

    monkeypatch.setattr(build_engine_script, "parse_onnx_network", fake_parse_onnx_network)
    monkeypatch.setattr(
        build_engine_script, "configure_builder_config", fake_configure_builder_config
    )

    with pytest.raises(RuntimeError, match="stop-before-real-build"):
        build_engine_script.build_engine(config, None, fp16_only=True, workspace_mb_override=256)

    assert captured["workspace_size_mb"] == 256
