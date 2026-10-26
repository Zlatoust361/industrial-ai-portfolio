"""Script-level tests for scripts/04_detect.py.

Covers the day 4 ask "проверь, что конфиг стадий действительно
переключается" (verify stage config actually switches) concretely: running
04_detect.py with and without --device jetson against the same demo data
and confirming device.batch_size's effect (forced to 1 on jetson, see
utils.config.apply_device_target) is visible in how images are actually
grouped into detect_batch() calls -- not just that apply_device_target
returns the right dataclass (already covered by tests/test_config.py).

Uses detect.backend: stub (StubDetector) throughout -- no heavy deps, so
this runs for real rather than stopping at an import boundary.

Run with (from pipeline/):
    PYTHONPATH=. pytest tests/test_script_04_detect.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from conftest import load_script_module, write_pipeline_config
from PIL import Image

detect = load_script_module("04_detect.py")

from utils.config import load_config  # noqa: E402


def _write_processed_manifest(tmp_path: Path, count: int) -> None:
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)
    images = []
    for i in range(count):
        path = processed_dir / f"img{i}.jpg"
        Image.new("RGB", (10, 10), color=(120, 120, 120)).save(path)
        images.append({"processed_path": str(path)})
    (tmp_path / "data" / "processed_manifest.json").write_text(
        json.dumps({"target_size": [224, 224], "images": images})
    )


def _spy_on_detect_batch(monkeypatch) -> list[int]:
    """Patch StubDetector.detect_batch to record each call's batch size."""
    batch_sizes_seen: list[int] = []
    original = detect.StubDetector.detect_batch

    def spy(self, image_paths):
        batch_sizes_seen.append(len(image_paths))
        return original(self, image_paths)

    monkeypatch.setattr(detect.StubDetector, "detect_batch", spy)
    return batch_sizes_seen


def test_chunk_splits_into_groups_of_configured_size() -> None:
    items = [Path(f"{i}.jpg") for i in range(5)]
    assert detect._chunk(items, 2) == [
        [Path("0.jpg"), Path("1.jpg")],
        [Path("2.jpg"), Path("3.jpg")],
        [Path("4.jpg")],
    ]


def test_chunk_empty_list_returns_empty() -> None:
    assert detect._chunk([], 3) == []


def test_build_detector_returns_stub_for_stub_backend(tmp_path: Path) -> None:
    config = load_config(write_pipeline_config(tmp_path, backend="stub"))
    assert isinstance(detect.build_detector(config), detect.StubDetector)


def test_build_detector_returns_clip_for_clip_backend(tmp_path: Path) -> None:
    config = load_config(write_pipeline_config(tmp_path, backend="clip"))
    assert isinstance(detect.build_detector(config), detect.ClipZeroShotDetector)


def test_build_detector_returns_yolo_for_yolo_backend(tmp_path: Path) -> None:
    config = load_config(write_pipeline_config(tmp_path, backend="yolo"))
    assert isinstance(detect.build_detector(config), detect.YoloDetector)


def test_main_batches_by_configured_batch_size(tmp_path: Path, monkeypatch) -> None:
    config_path = write_pipeline_config(tmp_path, backend="stub", batch_size=3)
    monkeypatch.chdir(tmp_path)
    _write_processed_manifest(tmp_path, count=5)
    batch_sizes_seen = _spy_on_detect_batch(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["04_detect.py", "--config", str(config_path)])

    detect.main()

    assert batch_sizes_seen == [3, 2]  # 5 images, batch_size=3 from config -> chunks of 3, 2


def test_main_device_jetson_forces_batch_size_one(tmp_path: Path, monkeypatch) -> None:
    config_path = write_pipeline_config(tmp_path, backend="stub", batch_size=3)
    monkeypatch.chdir(tmp_path)
    _write_processed_manifest(tmp_path, count=5)
    batch_sizes_seen = _spy_on_detect_batch(monkeypatch)
    monkeypatch.setattr(
        sys, "argv", ["04_detect.py", "--config", str(config_path), "--device", "jetson"]
    )

    detect.main()

    assert batch_sizes_seen == [1, 1, 1, 1, 1]  # --device jetson forces batch_size=1


def test_main_writes_detections_with_configured_backend(tmp_path: Path, monkeypatch) -> None:
    config_path = write_pipeline_config(tmp_path, backend="stub", batch_size=8)
    monkeypatch.chdir(tmp_path)
    _write_processed_manifest(tmp_path, count=2)
    monkeypatch.setattr(sys, "argv", ["04_detect.py", "--config", str(config_path)])

    detect.main()

    detections = json.loads((tmp_path / "outputs" / "detections.json").read_text())
    assert detections["backend"] == "stub"
    assert len(detections["detections"]) == 2
    assert all("defect_score" in entry for entry in detections["detections"])


def test_main_missing_processed_manifest_exits_with_error(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    config_path = write_pipeline_config(tmp_path, backend="stub")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["04_detect.py", "--config", str(config_path)])

    with pytest.raises(SystemExit) as exc_info:
        detect.main()

    assert exc_info.value.code == 1
    assert "03_preprocess.py" in capsys.readouterr().err
