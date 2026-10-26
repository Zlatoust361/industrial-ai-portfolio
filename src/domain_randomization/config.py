"""Typed loader for `config/parameters.yaml` (week 07: domain-randomized scene simulation).

Same "typed YAML section -> frozen dataclass" shape as `src/config.py`
(week 01), `pipeline/utils/config.py` (week 04), and
`src/synthetic/config.py` (week 05) -- kept as its own module since this
schema (camera/light/material/noise ranges for 2D scene compositing) has
nothing in common with any of those.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

FloatRange = tuple[float, float]


@dataclass(frozen=True)
class CanvasConfig:
    """`canvas:` section.

    Attributes:
        size: `(width, height)` of the rendered image, in pixels.
    """

    size: tuple[int, int]


@dataclass(frozen=True)
class CameraConfig:
    """`camera:` section -- object placement on the canvas.

    Attributes:
        scale_range: Multiplier applied to the object's native size.
        rotation_range_deg: In-plane rotation, in degrees.
        translation_fraction: Max center offset from the canvas center, as
            a fraction of canvas size (each axis sampled independently).
    """

    scale_range: FloatRange
    rotation_range_deg: FloatRange
    translation_fraction: float


@dataclass(frozen=True)
class LightConfig:
    """`light:` section -- global frame brightness/contrast/white-balance.

    Attributes:
        brightness_range: Multiplicative brightness factor.
        contrast_range: Multiplicative contrast factor.
        color_temperature_range: Warm(negative)/cool(positive) R/B channel tint strength.
    """

    brightness_range: FloatRange
    contrast_range: FloatRange
    color_temperature_range: FloatRange


@dataclass(frozen=True)
class MaterialConfig:
    """`material:` section -- the object's own surface appearance.

    Attributes:
        hue_shift_range_deg: Hue rotation, in degrees, on a 0-360 hue wheel.
        saturation_range: Multiplicative saturation factor.
    """

    hue_shift_range_deg: FloatRange
    saturation_range: FloatRange


@dataclass(frozen=True)
class NoiseConfig:
    """`noise:` section -- sensor-level degradation, applied last.

    Attributes:
        gaussian_std_range: Additive gaussian noise std, as a fraction of `[0, 255]`.
        blur_radius_range: Gaussian blur sigma, in pixels.
    """

    gaussian_std_range: FloatRange
    blur_radius_range: FloatRange


@dataclass(frozen=True)
class BackgroundConfig:
    """`background:` section -- solid fill used when no real background image is given.

    Attributes:
        color_low: `(r, g, b)` lower bound, each channel sampled independently.
        color_high: `(r, g, b)` upper bound, each channel sampled independently.
    """

    color_low: tuple[int, int, int]
    color_high: tuple[int, int, int]


@dataclass(frozen=True)
class DRParameters:
    """Root config object composed of every section in `parameters.yaml`."""

    seed: int
    canvas: CanvasConfig
    camera: CameraConfig
    light: LightConfig
    material: MaterialConfig
    noise: NoiseConfig
    background: BackgroundConfig


def load_dr_parameters(path: str | Path = "config/parameters.yaml") -> DRParameters:
    """Load and validate the scene-simulation config from a YAML file.

    Args:
        path: Path to a YAML file matching `parameters.yaml`'s schema.

    Returns:
        A fully parsed, immutable `DRParameters`.

    Raises:
        FileNotFoundError: If `path` does not exist.
        KeyError: If a required section or field is missing.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Parameters config file not found: {config_path}")

    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    canvas_raw = raw["canvas"]
    camera_raw = raw["camera"]
    light_raw = raw["light"]
    material_raw = raw["material"]
    noise_raw = raw["noise"]
    background_raw = raw["background"]

    return DRParameters(
        seed=int(raw["seed"]),
        canvas=CanvasConfig(size=tuple(canvas_raw["size"])),
        camera=CameraConfig(
            scale_range=tuple(camera_raw["scale_range"]),
            rotation_range_deg=tuple(camera_raw["rotation_range_deg"]),
            translation_fraction=float(camera_raw["translation_fraction"]),
        ),
        light=LightConfig(
            brightness_range=tuple(light_raw["brightness_range"]),
            contrast_range=tuple(light_raw["contrast_range"]),
            color_temperature_range=tuple(light_raw["color_temperature_range"]),
        ),
        material=MaterialConfig(
            hue_shift_range_deg=tuple(material_raw["hue_shift_range_deg"]),
            saturation_range=tuple(material_raw["saturation_range"]),
        ),
        noise=NoiseConfig(
            gaussian_std_range=tuple(noise_raw["gaussian_std_range"]),
            blur_radius_range=tuple(noise_raw["blur_radius_range"]),
        ),
        background=BackgroundConfig(
            color_low=tuple(background_raw["color_low"]),
            color_high=tuple(background_raw["color_high"]),
        ),
    )
