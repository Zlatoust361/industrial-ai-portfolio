"""Unit tests for `src.synthetic.config`."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.synthetic.config import ShapeSpec, load_defects_config


def test_load_defects_config_parses_the_real_config_file() -> None:
    config = load_defects_config("config/defects.yaml")

    assert config.seed == 42
    assert 0.0 < config.margin_fraction < 1.0
    assert config.count_range[0] <= config.count_range[1]
    assert {shape.type for shape in config.shapes} == {"line", "ellipse", "polyline"}


def test_load_defects_config_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_defects_config("config/does_not_exist.yaml")


def _write_config(tmp_path: Path, shapes_yaml: str) -> Path:
    config_path = tmp_path / "defects.yaml"
    config_path.write_text(
        f"seed: 1\nmargin_fraction: 0.05\ncount_range: [1, 2]\nshapes:\n{shapes_yaml}"
    )
    return config_path


def test_load_defects_config_rejects_unknown_shape_type(tmp_path: Path) -> None:
    path = _write_config(tmp_path, "  - name: blob\n    type: star\n    weight: 1.0\n")
    with pytest.raises(ValueError, match="star"):
        load_defects_config(path)


def test_load_defects_config_rejects_ellipse_missing_size_range(tmp_path: Path) -> None:
    path = _write_config(tmp_path, "  - name: pit\n    type: ellipse\n    weight: 1.0\n")
    with pytest.raises(ValueError, match="size_range"):
        load_defects_config(path)


def test_load_defects_config_rejects_line_missing_width_range(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path, "  - name: scratch\n    type: line\n    weight: 1.0\n    length_range: [10, 20]\n"
    )
    with pytest.raises(ValueError, match="width_range"):
        load_defects_config(path)


def test_load_defects_config_parses_a_valid_line_shape(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        "  - name: scratch\n    type: line\n    weight: 1.0\n"
        "    width_range: [2, 6]\n    length_range: [10, 20]\n",
    )
    config = load_defects_config(path)

    assert config.shapes == (
        ShapeSpec(
            name="scratch", type="line", weight=1.0, width_range=(2, 6), length_range=(10, 20)
        ),
    )
