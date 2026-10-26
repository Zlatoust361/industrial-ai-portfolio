"""Unit tests for `utils.config`."""

from __future__ import annotations

from pathlib import Path

import pytest
from utils.config import load_config

VALID_YAML = """
paths:
  data_dir: data
  outputs_dir: outputs
generation:
  count: 10
  defect_fraction: 0.5
  object_class_name: part
  defect_class_name: defect
  defect_overlay_color: [200, 40, 40]
  defect_overlay_alpha: 0.5
  object_pool_size: 3
  seed: 1
  dr_parameters_path: ../config/parameters.yaml
  defects_config_path: ../config/defects.yaml
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
  min_count: 200
  target_count: 250
  max_count: 300
  seed: 4
"""


def test_load_config_parses_every_section(tmp_path: Path) -> None:
    config_path = tmp_path / "dataset_config.yaml"
    config_path.write_text(VALID_YAML)

    config = load_config(config_path)

    assert config.paths.data_dir == Path("data")
    assert config.generation.count == 10
    assert config.generation.defect_overlay_color == (200, 40, 40)
    assert config.stratify.brightness_low_max == 0.9
    assert config.augment.augment_fraction == 0.3
    assert config.split.train_fraction == 0.7
    assert config.jetson_subset.target_count == 250


def test_load_config_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        load_config(tmp_path / "missing.yaml")


def test_load_config_rejects_split_fractions_not_summing_to_one(tmp_path: Path) -> None:
    bad_yaml = VALID_YAML.replace("test_fraction: 0.15", "test_fraction: 0.5")
    config_path = tmp_path / "dataset_config.yaml"
    config_path.write_text(bad_yaml)

    with pytest.raises(ValueError, match="sum to 1.0"):
        load_config(config_path)
