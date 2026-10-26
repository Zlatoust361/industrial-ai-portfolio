"""Unit tests for `utils/tensorrt_builder.py`'s non-TensorRT-dependent pieces.

`load_calibration_manifest`/`preprocess_calibration_batch` need only
stdlib/`numpy`/`PIL` -- both real dependencies of this sandbox -- so they
run for real here. `build_calibrator`/`parse_onnx_network`/
`configure_builder_config` need `tensorrt`/`pycuda` and are only exercised
up to that import boundary, via the script-level tests
(`test_script_04_quantize_model.py`/`test_script_05_build_engine.py`).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from utils.tensorrt_builder import load_calibration_manifest, preprocess_calibration_batch


def test_load_calibration_manifest_reads_output_paths(tmp_path: Path) -> None:
    calibration_dir = tmp_path / "calibration_data"
    calibration_dir.mkdir()
    (calibration_dir / "manifest.json").write_text(
        json.dumps({"images": [{"output": "a.png"}, {"output": "b.png"}]})
    )

    paths = load_calibration_manifest(calibration_dir)

    assert paths == [Path("a.png"), Path("b.png")]


def test_load_calibration_manifest_missing_raises_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="03_prepare_calibration_data"):
        load_calibration_manifest(tmp_path / "does_not_exist")


def test_preprocess_calibration_batch_shape_and_normalization(tmp_path: Path) -> None:
    image_path = tmp_path / "calib_0000.png"
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(image_path)

    batch = preprocess_calibration_batch([image_path], image_size=8)

    assert batch.shape == (1, 3, 8, 8)
    assert batch.dtype == np.float32
    # A pure-white pixel normalized by CLIP's mean/std should be well above zero.
    assert float(batch[0, 0, 0, 0]) > 0


def test_preprocess_calibration_batch_multiple_images(tmp_path: Path) -> None:
    paths = []
    for index in range(3):
        path = tmp_path / f"calib_{index:04d}.png"
        Image.new("RGB", (4, 4), color=(index * 50, 0, 0)).save(path)
        paths.append(path)

    batch = preprocess_calibration_batch(paths, image_size=4)

    assert batch.shape == (3, 3, 4, 4)
