"""Script-level tests for `scripts/03_prepare_calibration_data.py`.

Pure file I/O + PIL resizing -- no `torch`/`tensorrt` involved, so unlike
`01`/`02`/`04`/`05` this one runs for real end to end in this sandbox, not
just up to an import boundary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from conftest import load_script_module, write_config
from PIL import Image

prepare_calibration_data = load_script_module("03_prepare_calibration_data.py")


def _make_source_images(source_dir: Path, count: int) -> None:
    source_dir.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        Image.new("RGB", (32, 32), color=(index * 10 % 256, 0, 0)).save(
            source_dir / f"img_{index:03d}.jpg"
        )


def _run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *args: str) -> dict[str, object]:
    config_path = write_config(tmp_path, calibration_num_samples=3)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["03_prepare_calibration_data.py", "--config", str(config_path), *args],
    )
    prepare_calibration_data.main()
    return json.loads((tmp_path / "outputs" / "calibration_data" / "manifest.json").read_text())


def test_main_samples_and_resizes_real_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_dir = tmp_path / "calibration_source"
    _make_source_images(source_dir, count=10)

    manifest = _run(tmp_path, monkeypatch)

    assert manifest["num_candidates"] == 10
    assert manifest["num_samples"] == 3
    assert len(manifest["images"]) == 3
    for entry in manifest["images"]:
        saved = Path(entry["output"])
        assert saved.is_file()
        with Image.open(saved) as image:
            assert image.size == (224, 224)


def test_main_deterministic_with_same_seed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_dir = tmp_path / "calibration_source"
    _make_source_images(source_dir, count=10)

    manifest_a = _run(tmp_path, monkeypatch)
    sources_a = sorted(entry["source"] for entry in manifest_a["images"])

    manifest_b = _run(tmp_path, monkeypatch)
    sources_b = sorted(entry["source"] for entry in manifest_b["images"])

    assert sources_a == sources_b


def test_main_warns_when_fewer_candidates_than_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source_dir = tmp_path / "calibration_source"
    _make_source_images(source_dir, count=2)  # fewer than the requested 3

    manifest = _run(tmp_path, monkeypatch)

    assert manifest["num_samples"] == 2  # capped, not an error
    assert "Warning" in capsys.readouterr().err


def test_main_raises_clear_error_when_source_dir_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(FileNotFoundError, match="dataset_pipeline"):
        _run(tmp_path, monkeypatch)  # calibration_source/ was never created


def test_cli_source_dir_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real_source = tmp_path / "elsewhere"
    _make_source_images(real_source, count=5)

    manifest = _run(tmp_path, monkeypatch, "--source-dir", str(real_source))

    assert manifest["source_dir"] == str(real_source)
    assert manifest["num_candidates"] == 5
