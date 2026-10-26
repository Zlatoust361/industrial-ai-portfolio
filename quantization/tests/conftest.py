"""Shared fixtures/helpers for quantization/tests/.

`write_config` and `load_script_module` mirror `pipeline/tests/conftest.py`'s
helpers of the same shape/purpose, adapted to this week's own config schema.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"

_CONFIG_TEMPLATE = """
model:
  name: openai/clip-vit-base-patch32
  type: clip
  precision: fp16
  device: cpu
  seed: 42
  cache_dir: .cache/models
memory:
  memory_budget_mb: 6144
  device_total_mb: 8192
paths:
  outputs_dir: {outputs_dir}
  onnx_path: {outputs_dir}/onnx/image_encoder.onnx
  calibration_data_dir: {outputs_dir}/calibration_data
  calibration_cache_path: {outputs_dir}/calibration/int4_calibration.cache
  engine_path: {outputs_dir}/engine/image_encoder.engine
  fp16_engine_path: {outputs_dir}/engine/image_encoder_fp16.engine
baseline:
  image_size: {image_size}
  batch_sizes: [1, 4]
  warmup_iters: 1
  num_iters: 2
onnx:
  opset: 17
  dynamic_batch: true
  atol: 0.001
  rtol: 0.001
calibration:
  source_dir: {calibration_source_dir}
  num_samples: {calibration_num_samples}
  image_size: {image_size}
evaluation:
  test_set_dir: {evaluation_test_set_dir}
  num_samples: {evaluation_num_samples}
  warmup_iters: 1
  num_iters: 2
  min_cosine_similarity: {min_cosine_similarity}
  throttle_temp_c: {throttle_temp_c}
quantization:
  method: ptq
  bits: {bits}
  calibrator: entropy
  per_channel: true
active_profile: {active_profile}
profiles:
  dev:
    description: "Dev-machine profile, for --force smoke tests only."
    device: cuda
    batch_size: 1
    workspace_size_mb: 4096
    fp16_fallback: true
  orin_nano_8gb:
    description: "Sane defaults for a real Jetson Orin Nano 8GB build."
    device: cuda
    batch_size: 1
    workspace_size_mb: 1024
    fp16_fallback: true
"""


def write_config(
    tmp_path: Path,
    *,
    outputs_dir: str = "outputs",
    image_size: int = 224,
    calibration_source_dir: str = "calibration_source",
    calibration_num_samples: int = 4,
    evaluation_test_set_dir: str = "test_set",
    evaluation_num_samples: int = 4,
    min_cosine_similarity: float = 0.98,
    throttle_temp_c: float = 85.0,
    bits: int = 4,
    active_profile: str = "orin_nano_8gb",
) -> Path:
    """Write a minimal valid `config.yaml`-shaped file to `tmp_path`.

    Args:
        tmp_path: Directory to write `config.yaml` into.
        outputs_dir: `paths.outputs_dir` value (relative to `tmp_path`).
        image_size: `baseline.image_size`/`calibration.image_size` value.
        calibration_source_dir: `calibration.source_dir` value.
        calibration_num_samples: `calibration.num_samples` value.
        evaluation_test_set_dir: `evaluation.test_set_dir` value.
        evaluation_num_samples: `evaluation.num_samples` value.
        min_cosine_similarity: `evaluation.min_cosine_similarity` value.
        throttle_temp_c: `evaluation.throttle_temp_c` value.
        bits: `quantization.bits` value.
        active_profile: `active_profile` value.

    Returns:
        Path to the written config file.
    """
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        _CONFIG_TEMPLATE.format(
            outputs_dir=outputs_dir,
            image_size=image_size,
            calibration_source_dir=calibration_source_dir,
            calibration_num_samples=calibration_num_samples,
            evaluation_test_set_dir=evaluation_test_set_dir,
            evaluation_num_samples=evaluation_num_samples,
            min_cosine_similarity=min_cosine_similarity,
            throttle_temp_c=throttle_temp_c,
            bits=bits,
            active_profile=active_profile,
        )
    )
    return config_path


def load_script_module(filename: str) -> ModuleType:
    """Import a numbered `scripts/*.py` file as a module (its name isn't a valid identifier).

    Args:
        filename: Script filename under `quantization/scripts/`, e.g. `"01_baseline_inference.py"`.

    Returns:
        The imported module (running its top-level code, but not `main()` --
        every script guards that behind `if __name__ == "__main__":`).
    """
    module_name = filename.removesuffix(".py").replace(".", "_")
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS_DIR / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
