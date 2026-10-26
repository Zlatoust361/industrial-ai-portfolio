"""Typed loader for `config/dataset_config.yaml` (week 06).

Same "typed YAML section -> frozen dataclass" shape as `pipeline/utils/config.py`
(week 04), `src/domain_randomization/config.py` (week 07), etc. -- kept as
its own module since this schema (synthetic generation/stratify/augment/
split/Jetson-subset parameters) has nothing in common with any of those.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

IntRange = tuple[int, int]
FloatTriple = tuple[float, float, float]


@dataclass(frozen=True)
class PathsConfig:
    """`paths:` section.

    Attributes:
        data_dir: Root of stage-to-stage artifacts (`images/`, `manifest.json`).
        outputs_dir: Root of final outputs (`coco/`, `yolo/`, `jetson_val_subset/`).
    """

    data_dir: Path
    outputs_dir: Path


@dataclass(frozen=True)
class GenerationConfig:
    """`generation:` section -- stage 1.

    Attributes:
        count: Total samples to generate.
        defect_fraction: Probability a generated sample is labeled `"defect"`.
        object_class_name: Category name for the rendered object's bbox.
        defect_class_name: Category name for defect-shape bboxes.
        defect_overlay_color: `(r, g, b)` blended onto the frame where the
            generated defect mask is set.
        defect_overlay_alpha: Blend strength, `[0, 1]`.
        object_pool_size: Distinct synthesized placeholder object cutouts
            to cycle through (no real object photography involved).
        seed: Base seed; combined with each sample's index for
            per-sample determinism.
        dr_parameters_path: Path to week 07's `parameters.yaml`.
        defects_config_path: Path to week 05's `defects.yaml`.
        diffusion_config_path: Path to week 05's `diffusion.yaml`
            (inpainting backend + quality-gate config) -- only read by
            `01_generate_synthetic.py --source diffusion` (see
            `course/TECH-DEBT.md` C.1 / `course/DECISIONS.md` ADR-XXX).
            Optional -- defaults to `"../config/diffusion.yaml"` (same
            "week 05/07 config, reused as-is" convention as
            `dr_parameters_path`/`defects_config_path`) so existing
            `dataset_config.yaml` files that predate the `--source`
            flag don't need editing to keep parsing.
    """

    count: int
    defect_fraction: float
    object_class_name: str
    defect_class_name: str
    defect_overlay_color: tuple[int, int, int]
    defect_overlay_alpha: float
    object_pool_size: int
    seed: int
    dr_parameters_path: Path
    defects_config_path: Path
    diffusion_config_path: Path


@dataclass(frozen=True)
class StratifyConfig:
    """`stratify:` section -- stage 2 bucket thresholds.

    Attributes:
        brightness_low_max: See `utils.stratify.brightness_bucket`.
        brightness_high_min: See `utils.stratify.brightness_bucket`.
        rotation_left_max_deg: See `utils.stratify.angle_bucket`.
        rotation_right_min_deg: See `utils.stratify.angle_bucket`.
        background_dark_max_mean: See `utils.stratify.background_bucket`.
        balance_warn_ratio: An axis (label/brightness/angle/background)
            logs a warning when its min-count-bucket / max-count-bucket
            ratio drops below this, `[0, 1]` -- `1.0` would warn on any
            imbalance at all, `0.0` never warns.
    """

    brightness_low_max: float
    brightness_high_min: float
    rotation_left_max_deg: float
    rotation_right_min_deg: float
    background_dark_max_mean: float
    balance_warn_ratio: float


@dataclass(frozen=True)
class AugmentConfig:
    """`augment:` section -- stage 3."""

    augment_fraction: float
    seed: int


@dataclass(frozen=True)
class SplitConfig:
    """`split:` section -- stage 6.

    Attributes:
        train_fraction: Fraction of each stratum assigned to `train`.
        val_fraction: Fraction of each stratum assigned to `val`.
        test_fraction: Fraction of each stratum assigned to `test`.
        seed: Shuffle seed.
    """

    train_fraction: float
    val_fraction: float
    test_fraction: float
    seed: int


@dataclass(frozen=True)
class JetsonSubsetConfig:
    """`jetson_subset:` section -- stage 7.

    Attributes:
        min_count: Minimum acceptable subset size.
        target_count: Preferred subset size.
        max_count: Maximum acceptable subset size.
        seed: Sampling seed.
    """

    min_count: int
    target_count: int
    max_count: int
    seed: int


@dataclass(frozen=True)
class DatasetPipelineConfig:
    """Root config object composed of every section in `dataset_config.yaml`."""

    paths: PathsConfig
    generation: GenerationConfig
    stratify: StratifyConfig
    augment: AugmentConfig
    split: SplitConfig
    jetson_subset: JetsonSubsetConfig


def load_config(path: str | Path = "config/dataset_config.yaml") -> DatasetPipelineConfig:
    """Load and validate the dataset-pipeline config from a YAML file.

    Args:
        path: Path to a YAML file matching `dataset_config.yaml`'s schema.

    Returns:
        A fully parsed, immutable `DatasetPipelineConfig`.

    Raises:
        FileNotFoundError: If `path` does not exist.
        KeyError: If a required section or field is missing.
        ValueError: If `split`'s three fractions don't sum to ~1.0.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Dataset pipeline config file not found: {config_path}")

    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    paths_raw = raw["paths"]
    generation_raw = raw["generation"]
    stratify_raw = raw["stratify"]
    augment_raw = raw["augment"]
    split_raw = raw["split"]
    jetson_subset_raw = raw["jetson_subset"]

    split = SplitConfig(
        train_fraction=float(split_raw["train_fraction"]),
        val_fraction=float(split_raw["val_fraction"]),
        test_fraction=float(split_raw["test_fraction"]),
        seed=int(split_raw["seed"]),
    )
    fraction_sum = split.train_fraction + split.val_fraction + split.test_fraction
    if abs(fraction_sum - 1.0) > 1e-6:
        raise ValueError(
            "split's train/val/test fractions must sum to 1.0, "
            f"got {split.train_fraction} + {split.val_fraction} + {split.test_fraction} "
            f"= {fraction_sum}."
        )

    return DatasetPipelineConfig(
        paths=PathsConfig(
            data_dir=Path(paths_raw["data_dir"]),
            outputs_dir=Path(paths_raw["outputs_dir"]),
        ),
        generation=GenerationConfig(
            count=int(generation_raw["count"]),
            defect_fraction=float(generation_raw["defect_fraction"]),
            object_class_name=generation_raw["object_class_name"],
            defect_class_name=generation_raw["defect_class_name"],
            defect_overlay_color=tuple(generation_raw["defect_overlay_color"]),
            defect_overlay_alpha=float(generation_raw["defect_overlay_alpha"]),
            object_pool_size=int(generation_raw["object_pool_size"]),
            seed=int(generation_raw["seed"]),
            dr_parameters_path=Path(generation_raw["dr_parameters_path"]),
            defects_config_path=Path(generation_raw["defects_config_path"]),
            # Optional (see GenerationConfig.diffusion_config_path's docstring) --
            # only used by --source diffusion, defaulted so pre-existing
            # dataset_config.yaml files without this key keep parsing unchanged.
            diffusion_config_path=Path(
                generation_raw.get("diffusion_config_path", "../config/diffusion.yaml")
            ),
        ),
        stratify=StratifyConfig(
            brightness_low_max=float(stratify_raw["brightness_low_max"]),
            brightness_high_min=float(stratify_raw["brightness_high_min"]),
            rotation_left_max_deg=float(stratify_raw["rotation_left_max_deg"]),
            rotation_right_min_deg=float(stratify_raw["rotation_right_min_deg"]),
            background_dark_max_mean=float(stratify_raw["background_dark_max_mean"]),
            balance_warn_ratio=float(stratify_raw["balance_warn_ratio"]),
        ),
        augment=AugmentConfig(
            augment_fraction=float(augment_raw["augment_fraction"]),
            seed=int(augment_raw["seed"]),
        ),
        split=split,
        jetson_subset=JetsonSubsetConfig(
            min_count=int(jetson_subset_raw["min_count"]),
            target_count=int(jetson_subset_raw["target_count"]),
            max_count=int(jetson_subset_raw["max_count"]),
            seed=int(jetson_subset_raw["seed"]),
        ),
    )
