"""Script-level tests for scripts/03_preprocess.py.

tests/test_preprocess.py already covers `compute_letterbox_geometry`'s pure
math; this covers the actual script -- real PIL images letterboxed onto
`preprocess.image_size` and written to `data/processed/`, alongside a
correct `data/processed_manifest.json`.

Run with (from pipeline/):
    PYTHONPATH=. pytest tests/test_script_03_preprocess.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from conftest import load_script_module, write_pipeline_config
from PIL import Image

preprocess = load_script_module("03_preprocess.py")


def _write_manifest(tmp_path: Path, images: list[tuple[str, int, int]]) -> None:
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    entries = []
    for name, width, height in images:
        path = raw_dir / name
        Image.new("RGB", (width, height), color=(100, 100, 100)).save(path)
        entries.append({"path": str(path), "width": width, "height": height})
    (tmp_path / "data" / "manifest.json").write_text(
        json.dumps(
            {"raw_dir": str(raw_dir), "input_glob": "*.jpg", "images": entries, "skipped": []}
        )
    )


def test_main_letterboxes_images_to_configured_size(tmp_path: Path, monkeypatch) -> None:
    config_path = write_pipeline_config(tmp_path, image_size=(100, 100))
    monkeypatch.chdir(tmp_path)
    _write_manifest(tmp_path, [("wide.jpg", 400, 100)])
    monkeypatch.setattr(sys, "argv", ["03_preprocess.py", "--config", str(config_path)])

    preprocess.main()

    processed_path = tmp_path / "data" / "processed" / "wide.jpg"
    assert processed_path.is_file()
    with Image.open(processed_path) as image:
        assert image.size == (100, 100)  # canvas is always exactly the configured target size

    manifest = json.loads((tmp_path / "data" / "processed_manifest.json").read_text())
    assert manifest["target_size"] == [100, 100]
    entry = manifest["images"][0]
    assert entry["scale"] == pytest.approx(0.25)  # 100/400, the binding axis
    assert entry["offset"] == [0, 37]  # centered top/bottom: (100 - 400*0.25) // 2


def test_main_preserves_multiple_images_in_manifest_order(tmp_path: Path, monkeypatch) -> None:
    config_path = write_pipeline_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    _write_manifest(tmp_path, [("a.jpg", 50, 50), ("b.jpg", 200, 100)])
    monkeypatch.setattr(sys, "argv", ["03_preprocess.py", "--config", str(config_path)])

    preprocess.main()

    manifest = json.loads((tmp_path / "data" / "processed_manifest.json").read_text())
    assert [Path(entry["source_path"]).name for entry in manifest["images"]] == ["a.jpg", "b.jpg"]
    assert (tmp_path / "data" / "processed" / "a.jpg").is_file()
    assert (tmp_path / "data" / "processed" / "b.jpg").is_file()


def test_main_missing_manifest_exits_with_error(tmp_path: Path, monkeypatch, capsys) -> None:
    config_path = write_pipeline_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["03_preprocess.py", "--config", str(config_path)])

    with pytest.raises(SystemExit) as exc_info:
        preprocess.main()

    assert exc_info.value.code == 1
    assert "02_ingest.py" in capsys.readouterr().err
