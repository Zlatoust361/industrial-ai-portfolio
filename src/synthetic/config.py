"""Typed loader for `config/defects.yaml` (week 05: synthetic defect generation).

Same "typed YAML section -> frozen dataclass" shape as `src/config.py`
(week 01) and `pipeline/utils/config.py` (week 04), kept as its own module
rather than folded into either: `AppConfig` is about one CLIP checkpoint's
inference settings, `PipelineConfig` is about the 7 inspection-pipeline
stages -- neither schema has anything to do with defect-shape geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SHAPE_TYPES: tuple[str, ...] = ("line", "ellipse", "polyline")
"""Valid `shapes[].type` values (see `src.synthetic.mask_generation`)."""


@dataclass(frozen=True)
class ShapeSpec:
    """One `shapes[]` entry -- a defect shape template.

    Attributes:
        name: Shape identifier, e.g. `"scratch"` (used in
            `DefectInstance.shape_name` for traceability).
        type: One of `SHAPE_TYPES`; picks which renderer draws it.
        weight: Relative sampling weight among all configured shapes (not
            required to sum to 1 -- normalized at sample time).
        width_range: `(min, max)` stroke width in pixels. Required for
            `"line"`/`"polyline"`, unused for `"ellipse"`.
        length_range: `(min, max)` path length in pixels. Required for
            `"line"`/`"polyline"`, unused for `"ellipse"`.
        size_range: `(min, max)` axis length in pixels, sampled
            independently per axis. Required for `"ellipse"`, unused
            otherwise.
        segments_range: `(min, max)` number of straight sub-segments.
            Required for `"polyline"`, unused otherwise.
    """

    name: str
    type: str
    weight: float
    width_range: tuple[int, int] | None = None
    length_range: tuple[int, int] | None = None
    size_range: tuple[int, int] | None = None
    segments_range: tuple[int, int] | None = None


@dataclass(frozen=True)
class DefectsConfig:
    """Root config object composed of every section in `defects.yaml`.

    Attributes:
        seed: Default RNG seed for reproducible generation.
        margin_fraction: Fraction of image width/height kept clear of
            every shape's full extent, on each edge.
        count_range: `(min, max)` defect instances per generated mask,
            inclusive on both ends.
        shapes: Configured shape templates.
    """

    seed: int
    margin_fraction: float
    count_range: tuple[int, int]
    shapes: tuple[ShapeSpec, ...]


_REQUIRED_FIELDS_BY_TYPE: dict[str, tuple[str, ...]] = {
    "line": ("width_range", "length_range"),
    "ellipse": ("size_range",),
    "polyline": ("width_range", "length_range", "segments_range"),
}


def load_defects_config(path: str | Path = "config/defects.yaml") -> DefectsConfig:
    """Load and validate the defect-generation config from a YAML file.

    Args:
        path: Path to a YAML file matching `defects.yaml`'s schema.

    Returns:
        A fully parsed, immutable `DefectsConfig`.

    Raises:
        FileNotFoundError: If `path` does not exist.
        KeyError: If a required top-level or shape field is missing.
        ValueError: If a shape's `type` isn't one of `SHAPE_TYPES`, or a
            field required for that type is missing.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Defects config file not found: {config_path}")

    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    shapes = tuple(_parse_shape(shape_raw) for shape_raw in raw["shapes"])

    return DefectsConfig(
        seed=int(raw["seed"]),
        margin_fraction=float(raw["margin_fraction"]),
        count_range=tuple(raw["count_range"]),
        shapes=shapes,
    )


def _parse_shape(shape_raw: dict[str, Any]) -> ShapeSpec:
    """Parse and validate one `shapes[]` entry.

    Raises:
        KeyError: If `name`/`type`/`weight` is missing.
        ValueError: If `type` is unknown, or a type-required field is missing.
    """
    shape_type = shape_raw["type"]
    if shape_type not in SHAPE_TYPES:
        raise ValueError(f"Unknown shape type {shape_type!r}; expected one of {SHAPE_TYPES}.")

    for field in _REQUIRED_FIELDS_BY_TYPE[shape_type]:
        if field not in shape_raw:
            raise ValueError(
                f"Shape {shape_raw.get('name')!r} (type {shape_type!r}) missing required "
                f"field {field!r}."
            )

    def _range(field: str) -> tuple[int, int] | None:
        return tuple(shape_raw[field]) if field in shape_raw else None

    return ShapeSpec(
        name=shape_raw["name"],
        type=shape_type,
        weight=float(shape_raw["weight"]),
        width_range=_range("width_range"),
        length_range=_range("length_range"),
        size_range=_range("size_range"),
        segments_range=_range("segments_range"),
    )
