"""Typed loader for `config/scenarios.yaml` (week 12, day 3).

Same "typed YAML section -> frozen dataclass" shape as `src/config.py`/
`assistant/utils/config.py`/`src/robot/manipulator.py`'s own
`load_manipulator_config`. The one thing genuinely new here versus those:
each scenario's `camera:` section has an explicit `dev`/`jetson`
sub-entry -- resolved by `src.config.environment.Environment` at load
time, not left for each scenario to branch on itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.config.environment import Environment
from src.robot.manipulator import JointPosition


@dataclass(frozen=True)
class CameraSource:
    """One environment's camera source.

    Attributes:
        source: `"synthetic"` (dev -- a deterministic placeholder frame
            generator, no camera needed) or a real device path/RTSP URL
            (jetson -- passed straight to
            `src.data.video.open_video_capture`).
    """

    source: str

    @property
    def is_synthetic(self) -> bool:
        """`True` for the `"synthetic"` dev source; `False` for a real camera/RTSP URL."""
        return self.source == "synthetic"


@dataclass(frozen=True)
class ColorSortingConfig:
    """`color_sorting:` section of `config/scenarios.yaml`.

    Attributes:
        colors: Class labels the zero-shot classifier chooses among.
        margin_threshold: Minimum winning-vs-runner-up similarity margin
            (`ClassificationResult.margin`) to act on a classification;
            below this, the scenario reports "uncertain" rather than
            sorting into a possibly-wrong bin.
        pick_position: Where the arm picks an item up from.
        bin_positions: Color label -> drop-off position for that bin.
        camera: This environment's camera source.
    """

    colors: tuple[str, ...]
    margin_threshold: float
    pick_position: JointPosition
    bin_positions: dict[str, JointPosition]
    camera: CameraSource


@dataclass(frozen=True)
class DefectDetectionConfig:
    """`defect_detection:` section of `config/scenarios.yaml`.

    Attributes:
        margin_threshold: Same meaning as `ColorSortingConfig`'s.
        pick_position: Where the arm picks an item up from.
        reject_position: Where a defective item is placed.
        camera: This environment's camera source.
    """

    margin_threshold: float
    pick_position: JointPosition
    reject_position: JointPosition
    camera: CameraSource


def _load_camera_source(raw: dict[str, Any], environment: Environment) -> CameraSource:
    env_raw = raw["camera"][environment.value]
    return CameraSource(source=str(env_raw["source"]))


def load_color_sorting_config(
    path: str | Path = "config/scenarios.yaml", environment: Environment = Environment.DEV
) -> ColorSortingConfig:
    """Load and validate `color_sorting.py`'s config for one environment.

    Args:
        path: Path to a YAML file matching `config/scenarios.yaml`'s schema.
        environment: Which `camera:` sub-entry to resolve.

    Returns:
        A fully parsed, immutable `ColorSortingConfig`.

    Raises:
        FileNotFoundError: If `path` does not exist.
        KeyError: If a required field (including `bin_positions` missing
            one of `colors`) is missing.
        ValueError: If `margin_threshold` is negative.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Scenarios config file not found: {config_path}")
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))["color_sorting"]

    colors = tuple(str(c) for c in raw["colors"])
    bin_positions = {
        color: JointPosition.from_dict(raw["bin_positions"][color]) for color in colors
    }
    config = ColorSortingConfig(
        colors=colors,
        margin_threshold=float(raw["margin_threshold"]),
        pick_position=JointPosition.from_dict(raw["pick_position"]),
        bin_positions=bin_positions,
        camera=_load_camera_source(raw, environment),
    )
    if config.margin_threshold < 0:
        raise ValueError("margin_threshold must be >= 0.")
    return config


def load_defect_detection_config(
    path: str | Path = "config/scenarios.yaml", environment: Environment = Environment.DEV
) -> DefectDetectionConfig:
    """Load and validate `defect_detection.py`'s config for one environment.

    Args:
        path: Path to a YAML file matching `config/scenarios.yaml`'s schema.
        environment: Which `camera:` sub-entry to resolve.

    Returns:
        A fully parsed, immutable `DefectDetectionConfig`.

    Raises:
        FileNotFoundError: If `path` does not exist.
        KeyError: If a required field is missing.
        ValueError: If `margin_threshold` is negative.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Scenarios config file not found: {config_path}")
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))[
        "defect_detection"
    ]

    config = DefectDetectionConfig(
        margin_threshold=float(raw["margin_threshold"]),
        pick_position=JointPosition.from_dict(raw["pick_position"]),
        reject_position=JointPosition.from_dict(raw["reject_position"]),
        camera=_load_camera_source(raw, environment),
    )
    if config.margin_threshold < 0:
        raise ValueError("margin_threshold must be >= 0.")
    return config
