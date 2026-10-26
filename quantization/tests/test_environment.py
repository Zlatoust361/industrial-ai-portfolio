"""Unit tests for `utils/environment.py`."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from utils.environment import is_jetson, require_jetson


def test_is_jetson_true_when_marker_present(tmp_path: Path) -> None:
    marker = tmp_path / "nv_tegra_release"
    marker.write_text("# R36 (release), REVISION: 4.0\n")

    assert is_jetson(marker) is True


def test_is_jetson_false_when_marker_absent(tmp_path: Path) -> None:
    assert is_jetson(tmp_path / "no_such_marker") is False


def test_require_jetson_returns_when_marker_present(tmp_path: Path) -> None:
    marker = tmp_path / "nv_tegra_release"
    marker.write_text("# R36 (release), REVISION: 4.0\n")

    require_jetson(marker, "some_script.py", force=False)  # must not raise/exit


def test_require_jetson_returns_when_forced_even_without_marker(tmp_path: Path) -> None:
    require_jetson(tmp_path / "no_such_marker", "some_script.py", force=True)  # must not raise/exit


def test_require_jetson_exits_when_marker_absent_and_not_forced(tmp_path: Path) -> None:
    stream = io.StringIO()

    with pytest.raises(SystemExit) as exc_info:
        require_jetson(
            tmp_path / "no_such_marker", "05_build_engine.py", force=False, stream=stream
        )

    assert exc_info.value.code == 1
    assert "05_build_engine.py" in stream.getvalue()
    assert "Jetson" in stream.getvalue()
