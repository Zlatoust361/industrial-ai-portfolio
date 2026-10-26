"""Shared fixtures/helpers for `dataset_pipeline/tests/`.

`write_dataset_config` and `load_script_module` are used by the
script-level (`test_script_*.py`) tests, which exercise each stage's
`main()`/CLI wiring end to end (as opposed to the `test_*.py` files
alongside them, which test `utils/`'s pure logic in isolation) -- same
split as `pipeline/tests/conftest.py` (week 04).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DR_PARAMETERS_PATH = REPO_ROOT / "config" / "parameters.yaml"
DEFECTS_CONFIG_PATH = REPO_ROOT / "config" / "defects.yaml"
DIFFUSION_CONFIG_PATH = REPO_ROOT / "config" / "diffusion.yaml"

_CONFIG_TEMPLATE = """
paths:
  data_dir: {data_dir}
  outputs_dir: {outputs_dir}
generation:
  count: {count}
  defect_fraction: 0.5
  object_class_name: part
  defect_class_name: defect
  defect_overlay_color: [200, 40, 40]
  defect_overlay_alpha: 0.55
  object_pool_size: 2
  seed: 1
  dr_parameters_path: {dr_parameters_path}
  defects_config_path: {defects_config_path}
  diffusion_config_path: {diffusion_config_path}
stratify:
  brightness_low_max: 0.9
  brightness_high_min: 1.1
  rotation_left_max_deg: -5.0
  rotation_right_min_deg: 5.0
  background_dark_max_mean: 200.0
  balance_warn_ratio: 0.3
augment:
  augment_fraction: 0.3
  seed: 2
split:
  train_fraction: 0.7
  val_fraction: 0.15
  test_fraction: 0.15
  seed: 3
jetson_subset:
  min_count: {jetson_min_count}
  target_count: {jetson_target_count}
  max_count: {jetson_max_count}
  seed: 4
"""


def write_dataset_config(
    tmp_path: Path,
    *,
    count: int = 12,
    jetson_min_count: int = 2,
    jetson_target_count: int = 3,
    jetson_max_count: int = 4,
) -> Path:
    """Write a minimal valid `dataset_config.yaml`-shaped file under `tmp_path`.

    Uses the real, repo-root `config/parameters.yaml`/`config/defects.yaml`/
    `config/diffusion.yaml` (absolute paths, so this works regardless of
    cwd/`monkeypatch.chdir`) rather than a second test-only copy -- generation needs a real, valid
    DR/defects config either way, and this is the one the rest of the repo
    already maintains.

    Args:
        tmp_path: Directory the config's `paths.data_dir`/`outputs_dir`
            point into, and where the config file itself is written.
        count: `generation.count`, kept small for fast tests.
        jetson_min_count: `jetson_subset.min_count`, kept small so a
            small `count` can still produce enough val samples.
        jetson_target_count: `jetson_subset.target_count`.
        jetson_max_count: `jetson_subset.max_count`.

    Returns:
        Path to the written config file.
    """
    config_path = tmp_path / "dataset_config.yaml"
    config_path.write_text(
        _CONFIG_TEMPLATE.format(
            data_dir=tmp_path / "data",
            outputs_dir=tmp_path / "outputs",
            count=count,
            dr_parameters_path=DR_PARAMETERS_PATH,
            defects_config_path=DEFECTS_CONFIG_PATH,
            diffusion_config_path=DIFFUSION_CONFIG_PATH,
            jetson_min_count=jetson_min_count,
            jetson_target_count=jetson_target_count,
            jetson_max_count=jetson_max_count,
        )
    )
    return config_path


def load_script_module(filename: str) -> ModuleType:
    """Import a numbered `scripts/*.py` file as a module (its name isn't a valid identifier).

    Args:
        filename: Script filename under `dataset_pipeline/scripts/`, e.g.
            `"01_generate_synthetic.py"`.

    Returns:
        The imported module (its top-level code runs, but not `main()` --
        every script guards that behind `if __name__ == "__main__":`).
    """
    module_name = filename.removesuffix(".py").replace(".", "_")
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS_DIR / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
