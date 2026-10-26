"""Unit tests for `src.config.load_config`."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import apply_target_overrides, load_config


def test_load_config_parses_all_sections() -> None:
    config = load_config("config/model_config.yaml")

    assert config.model.name == "openai/clip-vit-base-patch32"
    assert config.model.precision in {"fp16", "fp32"}
    assert config.model.seed == 42
    assert isinstance(config.model.cache_dir, Path)

    assert config.inference.batch_size > 0
    assert config.inference.image_size > 0

    assert config.api.port > 0

    assert config.export.onnx_opset > 0
    assert isinstance(config.export.onnx_output_path, Path)
    assert config.export.onnx_atol > 0
    assert config.export.onnx_rtol > 0

    assert config.memory.memory_budget_mb == 6144
    assert config.memory.device_total_mb == 8192
    assert config.memory.memory_budget_mb < config.memory.device_total_mb

    assert config.benchmark.batch_sizes == [1, 4, 8]
    assert isinstance(config.benchmark.output_dir, Path)


def test_load_config_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("config/does_not_exist.yaml")


def test_apply_target_overrides_dev_is_a_no_op() -> None:
    config = load_config("config/model_config.yaml")
    assert apply_target_overrides(config, "dev") is config


def test_apply_target_overrides_jetson_forces_fp16() -> None:
    config = load_config("config/model_config.yaml")
    jetson_config = apply_target_overrides(config, "jetson")

    assert jetson_config.model.precision == "fp16"
    assert config.model.precision != "fp16"  # original config left untouched


def test_apply_target_overrides_unknown_target_raises() -> None:
    config = load_config("config/model_config.yaml")
    with pytest.raises(ValueError, match="edge"):
        apply_target_overrides(config, "edge")
