"""Unit tests for `utils/config.py`."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import write_config
from utils.config import load_config


def test_load_config_parses_every_section(tmp_path: Path) -> None:
    config_path = write_config(tmp_path)
    config = load_config(config_path)

    assert config.model.name == "openai/clip-vit-base-patch32"
    assert config.model.precision == "fp16"
    assert config.memory.memory_budget_mb == 6144
    assert config.paths.onnx_path == Path("outputs/onnx/image_encoder.onnx")
    assert config.baseline.batch_sizes == [1, 4]
    assert config.onnx.opset == 17
    assert config.calibration.num_samples == 4
    assert config.quantization.method == "ptq"
    assert config.quantization.bits == 4
    assert config.active_profile == "orin_nano_8gb"


def test_load_config_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does_not_exist.yaml")


def test_load_config_rejects_unknown_quantization_method(tmp_path: Path) -> None:
    config_path = write_config(tmp_path)
    config_path.write_text(config_path.read_text().replace("method: ptq", "method: qat"))

    with pytest.raises(ValueError, match="quantization.method"):
        load_config(config_path)


def test_load_config_rejects_unknown_bits(tmp_path: Path) -> None:
    config_path = write_config(tmp_path, bits=6)

    with pytest.raises(ValueError, match="quantization.bits"):
        load_config(config_path)


def test_load_config_rejects_unknown_active_profile(tmp_path: Path) -> None:
    config_path = write_config(tmp_path)
    config_path.write_text(
        config_path.read_text().replace("active_profile: orin_nano_8gb", "active_profile: nope")
    )

    with pytest.raises(ValueError, match="active_profile"):
        load_config(config_path)


def test_get_profile_defaults_to_active_profile(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, active_profile="dev"))

    profile = config.get_profile()

    assert profile.workspace_size_mb == 4096  # dev's value, not orin_nano_8gb's


def test_get_profile_explicit_name_overrides_active_profile(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, active_profile="dev"))

    profile = config.get_profile("orin_nano_8gb")

    assert profile.batch_size == 1
    assert profile.workspace_size_mb == 1024
    assert profile.fp16_fallback is True


def test_get_profile_unknown_name_raises(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))

    with pytest.raises(KeyError, match="Unknown profile"):
        config.get_profile("does_not_exist")


def test_orin_nano_8gb_profile_matches_documented_defaults(tmp_path: Path) -> None:
    """The exact ask from week 09 day 2: batch=1, a bounded workspace, fp16 fallback on."""
    config = load_config(write_config(tmp_path))

    profile = config.get_profile("orin_nano_8gb")

    assert profile.batch_size == 1
    assert profile.workspace_size_mb < config.memory.memory_budget_mb
    assert profile.fp16_fallback is True
