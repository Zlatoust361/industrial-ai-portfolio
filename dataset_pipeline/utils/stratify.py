"""Stratum assignment + balance reporting from a sample's actual DR render
parameters (week 06, stage 2).

A stratum joins a sample's `label` with three condition buckets --
brightness/angle/background -- discretized from the DR renderer's *actual
per-sample* parameters (`Sample.dr_params`, stage 1's output), not the
config's sampling ranges. Stages 6/7 both stratified-sample using this
field.

`compute_distribution_report` (day 4) additionally computes each axis's
*marginal* distribution (label alone, brightness alone, ...) -- the joint
`stratum_distribution` groups by all four together, which is what
stages 6/7 need for sampling, but doesn't answer "is brightness alone
balanced?" on its own (e.g. 30 single-sample joint strata could still add
up to a badly skewed marginal brightness split, or vice versa) --
`02_stratify_dataset.py` writes the combined report to
`outputs/metrics/stratify_distribution.json`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from utils.config import StratifyConfig
from utils.manifest import Sample, with_stratum

LABELS: tuple[str, ...] = ("ok", "defect")
BRIGHTNESS_BUCKETS: tuple[str, ...] = ("low", "mid", "high")
ANGLE_BUCKETS: tuple[str, ...] = ("left", "center", "right")
BACKGROUND_BUCKETS: tuple[str, ...] = ("dark", "light")


def brightness_bucket(brightness: float, config: StratifyConfig) -> str:
    """Discretize `light.brightness` into `"low"`/`"mid"`/`"high"`.

    Args:
        brightness: A sample's `dr_params["light"]["brightness"]`.
        config: Stratify thresholds.

    Returns:
        `"low"` if below `brightness_low_max`, `"high"` if at/above
        `brightness_high_min`, else `"mid"`.
    """
    if brightness < config.brightness_low_max:
        return "low"
    if brightness >= config.brightness_high_min:
        return "high"
    return "mid"


def angle_bucket(rotation_deg: float, config: StratifyConfig) -> str:
    """Discretize `camera.rotation_deg` into `"left"`/`"center"`/`"right"`.

    Args:
        rotation_deg: A sample's `dr_params["camera"]["rotation_deg"]`.
        config: Stratify thresholds.

    Returns:
        `"left"` if below `rotation_left_max_deg`, `"right"` if at/above
        `rotation_right_min_deg`, else `"center"`.
    """
    if rotation_deg < config.rotation_left_max_deg:
        return "left"
    if rotation_deg >= config.rotation_right_min_deg:
        return "right"
    return "center"


def background_bucket(background_color: tuple[int, int, int], config: StratifyConfig) -> str:
    """Discretize `background_color` into `"dark"`/`"light"` by mean channel value.

    Args:
        background_color: A sample's `dr_params["background_color"]`, `(r, g, b)`.
        config: Stratify thresholds.

    Returns:
        `"dark"` if the mean of the three channels is below
        `background_dark_max_mean`, else `"light"`.
    """
    mean_value = sum(background_color) / len(background_color)
    return "dark" if mean_value < config.background_dark_max_mean else "light"


def assign_stratum(sample: Sample, config: StratifyConfig) -> str:
    """Compute one sample's stratum string from its `dr_params` and `label`.

    Args:
        sample: A stage-1 sample (`dr_params` populated, `stratum` not yet set).
        config: Stratify thresholds.

    Returns:
        `"{label}_{brightness}_{angle}_{background}"`, e.g. `"defect_low_left_dark"`.
    """
    light = sample.dr_params["light"]
    camera = sample.dr_params["camera"]
    background_color = tuple(sample.dr_params["background_color"])

    return "_".join(
        (
            sample.label,
            brightness_bucket(light["brightness"], config),
            angle_bucket(camera["rotation_deg"], config),
            background_bucket(background_color, config),
        )
    )


def stratify_samples(samples: list[Sample], config: StratifyConfig) -> list[Sample]:
    """Assign a `stratum` to every sample.

    Args:
        samples: Stage-1 samples.
        config: Stratify thresholds.

    Returns:
        Copies of `samples` with `stratum` set, same order.
    """
    return [with_stratum(sample, assign_stratum(sample, config)) for sample in samples]


def stratum_distribution(samples: list[Sample]) -> dict[str, int]:
    """Count samples per stratum, most common first.

    Args:
        samples: Samples with `stratum` already assigned.

    Returns:
        `{stratum: count}`, ordered by descending count.
    """
    return dict(Counter(sample.stratum for sample in samples).most_common())


def _zero_filled_distribution(values: list[str], known_buckets: tuple[str, ...]) -> dict[str, int]:
    """Count `values` per bucket, explicitly zero-filling any `known_buckets` never observed.

    A bucket that never appears at all is a stronger imbalance signal than
    one that's merely rare -- `Counter.most_common()` alone would just
    omit it, silently hiding that from `balance_ratio` (whose `min()`
    needs the `0` to be present to detect it).
    """
    counts = Counter(values)
    return {bucket: counts.get(bucket, 0) for bucket in known_buckets}


def label_distribution(samples: list[Sample]) -> dict[str, int]:
    """Count samples per `label` (`"ok"`/`"defect"`), zero-filling any label never observed."""
    return _zero_filled_distribution([sample.label for sample in samples], LABELS)


def brightness_distribution(samples: list[Sample], config: StratifyConfig) -> dict[str, int]:
    """Count samples per brightness bucket, zero-filling any bucket never observed."""
    values = [
        brightness_bucket(sample.dr_params["light"]["brightness"], config) for sample in samples
    ]
    return _zero_filled_distribution(values, BRIGHTNESS_BUCKETS)


def angle_distribution(samples: list[Sample], config: StratifyConfig) -> dict[str, int]:
    """Count samples per angle bucket, zero-filling any bucket never observed."""
    values = [
        angle_bucket(sample.dr_params["camera"]["rotation_deg"], config) for sample in samples
    ]
    return _zero_filled_distribution(values, ANGLE_BUCKETS)


def background_distribution(samples: list[Sample], config: StratifyConfig) -> dict[str, int]:
    """Count samples per background bucket, zero-filling any bucket never observed."""
    values = [
        background_bucket(tuple(sample.dr_params["background_color"]), config) for sample in samples
    ]
    return _zero_filled_distribution(values, BACKGROUND_BUCKETS)


def balance_ratio(distribution: dict[str, int]) -> float:
    """The smallest bucket's share of the largest bucket's count, `[0, 1]`.

    `1.0` means every bucket has the same count (perfectly balanced);
    `0.0` means at least one known bucket was never observed at all
    (assuming `distribution` was zero-filled -- see
    `_zero_filled_distribution`) while another has samples.

    Args:
        distribution: `{bucket: count}`.

    Returns:
        `min(counts) / max(counts)`, or `1.0` for an empty distribution or
        one where every bucket is empty (nothing to be imbalanced about).
    """
    counts = list(distribution.values())
    if not counts or max(counts) == 0:
        return 1.0
    return min(counts) / max(counts)


@dataclass(frozen=True)
class DistributionReport:
    """Stage 2's full balance report: marginal (per-axis) and joint distributions.

    Attributes:
        total_samples: Number of samples the report was computed over.
        label_distribution: `{"ok"/"defect": count}`.
        brightness_distribution: `{"low"/"mid"/"high": count}`.
        angle_distribution: `{"left"/"center"/"right": count}`.
        background_distribution: `{"dark"/"light": count}`.
        stratum_distribution: The joint `label x brightness x angle x
            background` distribution (see `stratum_distribution`).
        balance_ratios: `{"label"/"brightness"/"angle"/"background":
            balance_ratio(...)}`.
        warnings: One message per axis whose `balance_ratios` entry is
            below the configured `balance_warn_ratio`; empty if every
            axis is acceptably balanced.
    """

    total_samples: int
    label_distribution: dict[str, int]
    brightness_distribution: dict[str, int]
    angle_distribution: dict[str, int]
    background_distribution: dict[str, int]
    stratum_distribution: dict[str, int]
    balance_ratios: dict[str, float]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """This report as a plain JSON-serializable dict."""
        return {
            "total_samples": self.total_samples,
            "label_distribution": self.label_distribution,
            "brightness_distribution": self.brightness_distribution,
            "angle_distribution": self.angle_distribution,
            "background_distribution": self.background_distribution,
            "stratum_distribution": self.stratum_distribution,
            "balance_ratios": self.balance_ratios,
            "warnings": list(self.warnings),
        }


def compute_distribution_report(
    samples: list[Sample], config: StratifyConfig
) -> DistributionReport:
    """Compute every marginal/joint distribution and balance ratio for `samples`.

    Args:
        samples: Samples with `stratum` already assigned (see `stratify_samples`).
        config: Stratify thresholds, including `balance_warn_ratio`.

    Returns:
        The full `DistributionReport`.
    """
    label_dist = label_distribution(samples)
    brightness_dist = brightness_distribution(samples, config)
    angle_dist = angle_distribution(samples, config)
    background_dist = background_distribution(samples, config)

    balance_ratios = {
        "label": balance_ratio(label_dist),
        "brightness": balance_ratio(brightness_dist),
        "angle": balance_ratio(angle_dist),
        "background": balance_ratio(background_dist),
    }
    axis_distributions = {
        "label": label_dist,
        "brightness": brightness_dist,
        "angle": angle_dist,
        "background": background_dist,
    }
    warnings = tuple(
        f"{axis} distribution is imbalanced: ratio {balance_ratios[axis]:.2f} "
        f"< {config.balance_warn_ratio:.2f} threshold ({axis_distributions[axis]})"
        for axis in ("label", "brightness", "angle", "background")
        if balance_ratios[axis] < config.balance_warn_ratio
    )

    return DistributionReport(
        total_samples=len(samples),
        label_distribution=label_dist,
        brightness_distribution=brightness_dist,
        angle_distribution=angle_dist,
        background_distribution=background_dist,
        stratum_distribution=stratum_distribution(samples),
        balance_ratios=balance_ratios,
        warnings=warnings,
    )
