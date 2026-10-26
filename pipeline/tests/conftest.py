"""Shared fixtures/helpers for pipeline/tests/.

`write_pipeline_config` and `load_script_module` are used by the
script-level (`test_script_*.py`) tests, which exercise each stage's
`main()`/CLI wiring end to end (as opposed to the `test_*.py` files
alongside them, which test `utils/`'s pure logic in isolation).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"

_CONFIG_TEMPLATE = """
paths:
  data_dir: data
  outputs_dir: outputs
  logs_dir: logs
device:
  target: dev
  precision: fp32
  batch_size: {batch_size}
  torch_device: cpu
  seed: 42
  memory_budget_mb: 6144
  device_total_mb: 8192
ingest:
  input_glob: "*.jpg"
preprocess:
  image_size: [{image_width}, {image_height}]
detect:
  backend: {backend}
  clip_model_name: openai/clip-vit-base-patch32
  clip_cache_dir: .cache
  clip_prompt_style: basic
  clip_prompts_path: ../config/prompts.yaml
  yolo_model_path: yolov8n.pt
  yolo_defect_classes: []
  score_threshold: 0.5
interpret:
  defect_score_threshold: {defect_score_threshold}
quality_gate:
  green_max_defect_rate: {green_max_defect_rate}
  yellow_max_defect_rate: {yellow_max_defect_rate}
report:
  title: "Inspection pipeline report"
"""


def write_pipeline_config(
    tmp_path: Path,
    *,
    backend: str = "stub",
    batch_size: int = 8,
    image_size: tuple[int, int] = (224, 224),
    defect_score_threshold: float = 0.5,
    green_max_defect_rate: float = 0.02,
    yellow_max_defect_rate: float = 0.10,
) -> Path:
    """Write a minimal valid `pipeline_config.yaml`-shaped file to `tmp_path`.

    Args:
        tmp_path: Directory to write `pipeline_config.yaml` into.
        backend: `detect.backend` value.
        batch_size: `device.batch_size` value.
        image_size: `preprocess.image_size` value.
        defect_score_threshold: `interpret.defect_score_threshold` value.
        green_max_defect_rate: `quality_gate.green_max_defect_rate` value.
        yellow_max_defect_rate: `quality_gate.yellow_max_defect_rate` value.

    Returns:
        Path to the written config file.
    """
    config_path = tmp_path / "pipeline_config.yaml"
    config_path.write_text(
        _CONFIG_TEMPLATE.format(
            backend=backend,
            batch_size=batch_size,
            image_width=image_size[0],
            image_height=image_size[1],
            defect_score_threshold=defect_score_threshold,
            green_max_defect_rate=green_max_defect_rate,
            yellow_max_defect_rate=yellow_max_defect_rate,
        )
    )
    return config_path


def load_script_module(filename: str) -> ModuleType:
    """Import a numbered `scripts/*.py` file as a module (its name isn't a valid identifier).

    Args:
        filename: Script filename under `pipeline/scripts/`, e.g. `"01_environment_check.py"`.

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
