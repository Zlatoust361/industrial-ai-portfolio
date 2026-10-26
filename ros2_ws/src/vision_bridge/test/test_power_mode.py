"""Unit tests for vision_bridge.power_mode.

No rclpy/colcon build needed -- pure Python. Run directly with:
    cd ros2_ws/src/vision_bridge && PYTHONPATH=. pytest test/test_power_mode.py
"""

from vision_bridge.power_mode import (
    SUFFICIENT_POWER_MODES,
    is_power_mode_sufficient,
    parse_nvpmodel_mode_name,
)


def test_parse_nvpmodel_mode_name_extracts_name() -> None:
    assert parse_nvpmodel_mode_name("NV Power Mode: MAXN_SUPER\n0\n") == "MAXN_SUPER"


def test_parse_nvpmodel_mode_name_handles_15w() -> None:
    assert parse_nvpmodel_mode_name("NV Power Mode: 15W\n1\n") == "15W"


def test_parse_nvpmodel_mode_name_handles_unexpected_output() -> None:
    assert parse_nvpmodel_mode_name("command not found") is None


def test_parse_nvpmodel_mode_name_empty_output() -> None:
    assert parse_nvpmodel_mode_name("") is None


def test_is_power_mode_sufficient_true_for_15w_and_maxn() -> None:
    assert is_power_mode_sufficient("15W") is True
    assert is_power_mode_sufficient("MAXN_SUPER") is True


def test_is_power_mode_sufficient_false_for_7w() -> None:
    assert is_power_mode_sufficient("7W") is False


def test_is_power_mode_sufficient_false_for_unknown_mode() -> None:
    assert is_power_mode_sufficient("SOME_OTHER_MODE") is False


def test_sufficient_power_modes_excludes_7w() -> None:
    assert "7W" not in SUFFICIENT_POWER_MODES
