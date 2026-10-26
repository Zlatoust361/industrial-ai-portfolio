"""Unit tests for utils/config.py.

Run with (from pipeline/):
    PYTHONPATH=. pytest tests/test_config.py
"""

from pathlib import Path

import pytest
from utils.config import apply_device_target, load_config

VALID_CONFIG_YAML = """
paths:
  data_dir: data
  outputs_dir: outputs
  logs_dir: logs
device:
  target: dev
  precision: fp32
  batch_size: 8
  torch_device: cpu
  seed: 42
  memory_budget_mb: 6144
  device_total_mb: 8192
ingest:
  input_glob: "*.jpg"
preprocess:
  image_size: [224, 224]
detect:
  backend: clip
  clip_model_name: openai/clip-vit-base-patch32
  clip_cache_dir: .cache
  clip_prompt_style: basic
  clip_prompts_path: ../config/prompts.yaml
  yolo_model_path: yolov8n.pt
  yolo_defect_classes: []
  score_threshold: 0.5
interpret:
  defect_score_threshold: 0.5
quality_gate:
  green_max_defect_rate: 0.02
  yellow_max_defect_rate: 0.10
report:
  title: "Inspection pipeline report"
"""


def _write_config(tmp_path: Path, yaml_text: str = VALID_CONFIG_YAML) -> Path:
    config_path = tmp_path / "pipeline_config.yaml"
    config_path.write_text(yaml_text)
    return config_path


def test_load_config_parses_every_section(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path))

    assert config.paths.data_dir == Path("data")
    assert config.device.target == "dev"
    assert config.device.precision == "fp32"
    assert config.device.batch_size == 8
    assert config.ingest.input_glob == "*.jpg"
    assert config.preprocess.image_size == (224, 224)
    assert config.detect.backend == "clip"
    assert config.detect.yolo_defect_classes == ()
    assert config.interpret.defect_score_threshold == 0.5
    assert config.quality_gate.green_max_defect_rate == 0.02
    assert config.report.title == "Inspection pipeline report"


def test_load_config_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/pipeline_config.yaml")


def test_load_config_rejects_unknown_device_target(tmp_path: Path) -> None:
    bad_yaml = VALID_CONFIG_YAML.replace("target: dev", "target: raspberry_pi")
    with pytest.raises(ValueError, match="device.target"):
        load_config(_write_config(tmp_path, bad_yaml))


def test_load_config_rejects_unknown_detect_backend(tmp_path: Path) -> None:
    bad_yaml = VALID_CONFIG_YAML.replace("backend: clip", "backend: magic")
    with pytest.raises(ValueError, match="detect.backend"):
        load_config(_write_config(tmp_path, bad_yaml))


def test_apply_device_target_dev_is_noop(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path))
    assert apply_device_target(config, "dev") is config


def test_apply_device_target_jetson_forces_fp16_and_batch_one(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path))
    jetson_config = apply_device_target(config, "jetson")

    assert jetson_config.device.target == "jetson"
    assert jetson_config.device.precision == "fp16"
    assert jetson_config.device.batch_size == 1
    # Original config is untouched (frozen dataclass -> a new copy is returned).
    assert config.device.precision == "fp32"
    assert config.device.batch_size == 8


def test_apply_device_target_rejects_unknown_target(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path))
    with pytest.raises(ValueError, match="Unknown device target"):
        apply_device_target(config, "raspberry_pi")
