"""Unit tests for utils/environment.py.

`build_environment_report`'s `available_memory_mb()` call is real `psutil`
(a real dependency, not mocked -- this sandbox has it, same as
`ros2_ws/scripts/resource_monitor.py`'s tests). `query_power_mode()` is left
unmocked too: this sandbox genuinely has no `nvpmodel` binary, so it
reliably returns `None` here, same as it would on any non-Jetson dev
machine -- day 4's roadmap prompt explicitly asks for a *mocked*
"we are on a Jetson" test, which belongs there once the mocking harness is
built; this file covers the pure parsing functions on their own.

Run with (from pipeline/):
    PYTHONPATH=. pytest tests/test_environment.py
"""

from pathlib import Path

from utils.environment import (
    build_environment_report,
    is_jetson,
    jetpack_generation_from_l4t,
    parse_l4t_version,
    parse_power_mode,
)

SAMPLE_NV_TEGRA_RELEASE = (
    "# R36 (release), REVISION: 4.0, GCID: 38968081, BOARD: generic, "
    "EABI: aarch64, DATE: Fri Jun 14 00:00:00 UTC 2024\n"
)


def test_is_jetson_true_when_marker_file_exists(tmp_path: Path) -> None:
    marker = tmp_path / "nv_tegra_release"
    marker.write_text(SAMPLE_NV_TEGRA_RELEASE)
    assert is_jetson(marker) is True


def test_is_jetson_false_when_marker_file_missing(tmp_path: Path) -> None:
    assert is_jetson(tmp_path / "does_not_exist") is False


def test_parse_l4t_version_extracts_release_and_revision() -> None:
    assert parse_l4t_version(SAMPLE_NV_TEGRA_RELEASE) == "R36.4.0"


def test_parse_l4t_version_no_match_returns_none() -> None:
    assert parse_l4t_version("garbage output") is None


def test_jetpack_generation_from_l4t_known_major_version() -> None:
    assert jetpack_generation_from_l4t("R36.4.0") == "6.x"
    assert jetpack_generation_from_l4t("R35.4.1") == "5.x"


def test_jetpack_generation_from_l4t_unknown_major_version_is_none() -> None:
    assert jetpack_generation_from_l4t("R99.0.0") is None


def test_jetpack_generation_from_l4t_none_input_is_none() -> None:
    assert jetpack_generation_from_l4t(None) is None


def test_parse_power_mode_extracts_name() -> None:
    assert parse_power_mode("NV Power Mode: 15W\n1\n") == "15W"


def test_parse_power_mode_no_match_returns_none() -> None:
    assert parse_power_mode("") is None


def test_build_environment_report_off_jetson(tmp_path: Path) -> None:
    report = build_environment_report(marker_path=tmp_path / "does_not_exist")

    assert report.is_jetson is False
    assert report.l4t_version is None
    assert report.jetpack_generation is None
    assert report.available_memory_mb > 0
    assert report.power_mode is None  # no nvpmodel binary in this sandbox


def test_build_environment_report_on_jetson(tmp_path: Path) -> None:
    marker = tmp_path / "nv_tegra_release"
    marker.write_text(SAMPLE_NV_TEGRA_RELEASE)

    report = build_environment_report(marker_path=marker)

    assert report.is_jetson is True
    assert report.l4t_version == "R36.4.0"
    assert report.jetpack_generation == "6.x"
    assert report.available_memory_mb > 0
