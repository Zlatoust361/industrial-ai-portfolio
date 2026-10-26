"""Tests for `src.scenarios.config` (week 12, day 3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config.environment import Environment
from src.scenarios.config import (
    CameraSource,
    load_color_sorting_config,
    load_defect_detection_config,
)


def test_load_color_sorting_config_parses_the_real_config_file() -> None:
    config = load_color_sorting_config("config/scenarios.yaml", environment=Environment.DEV)

    assert set(config.colors) == {"red", "green", "blue", "yellow"}
    assert set(config.bin_positions) == set(config.colors)
    assert config.margin_threshold >= 0
    assert config.camera.is_synthetic is True


def test_load_color_sorting_config_resolves_jetson_camera() -> None:
    config = load_color_sorting_config("config/scenarios.yaml", environment=Environment.JETSON)
    assert config.camera.is_synthetic is False
    assert config.camera.source == "/dev/video0"


def test_load_color_sorting_config_defaults_to_dev_environment() -> None:
    config = load_color_sorting_config("config/scenarios.yaml")
    assert config.camera.is_synthetic is True


def test_load_color_sorting_config_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_color_sorting_config("config/does_not_exist.yaml")


def test_load_color_sorting_config_rejects_negative_margin_threshold(tmp_path: Path) -> None:
    path = tmp_path / "scenarios.yaml"
    path.write_text(
        "color_sorting:\n"
        "  colors: [red]\n"
        "  margin_threshold: -1\n"
        "  pick_position: {shoulder_pan: 0, shoulder_lift: 0, elbow_flex: 0, "
        "wrist_flex: 0, wrist_roll: 0, gripper: 0}\n"
        "  bin_positions:\n"
        "    red: {shoulder_pan: 0, shoulder_lift: 0, elbow_flex: 0, "
        "wrist_flex: 0, wrist_roll: 0, gripper: 0}\n"
        "  camera:\n"
        "    dev: {source: synthetic}\n"
        "    jetson: {source: /dev/video0}\n"
    )
    with pytest.raises(ValueError, match="margin_threshold"):
        load_color_sorting_config(path)


def test_load_defect_detection_config_parses_the_real_config_file() -> None:
    config = load_defect_detection_config("config/scenarios.yaml", environment=Environment.DEV)

    assert config.margin_threshold >= 0
    assert config.camera.is_synthetic is True


def test_load_defect_detection_config_resolves_jetson_camera() -> None:
    config = load_defect_detection_config("config/scenarios.yaml", environment=Environment.JETSON)
    assert config.camera.is_synthetic is False


def test_camera_source_is_synthetic() -> None:
    assert CameraSource(source="synthetic").is_synthetic is True
    assert CameraSource(source="/dev/video0").is_synthetic is False
