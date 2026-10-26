"""Unit tests for `utils.stratify`."""

from __future__ import annotations

import json

import pytest
from utils.config import StratifyConfig
from utils.manifest import Sample
from utils.stratify import (
    angle_bucket,
    angle_distribution,
    assign_stratum,
    background_bucket,
    background_distribution,
    balance_ratio,
    brightness_bucket,
    brightness_distribution,
    compute_distribution_report,
    label_distribution,
    stratify_samples,
    stratum_distribution,
)

CONFIG = StratifyConfig(
    brightness_low_max=0.9,
    brightness_high_min=1.1,
    rotation_left_max_deg=-5.0,
    rotation_right_min_deg=5.0,
    background_dark_max_mean=200.0,
    balance_warn_ratio=0.3,
)


def _make_sample(
    brightness: float,
    rotation_deg: float,
    background: tuple[int, int, int],
    label: str = "ok",
) -> Sample:
    return Sample(
        sample_id="s",
        image_path="images/s.jpg",
        width=64,
        height=64,
        label=label,
        object_bbox=(0, 0, 10, 10),
        defect_boxes=(),
        dr_params={
            "light": {"brightness": brightness},
            "camera": {"rotation_deg": rotation_deg},
            "background_color": list(background),
        },
    )


def test_brightness_bucket_low_mid_high() -> None:
    assert brightness_bucket(0.7, CONFIG) == "low"
    assert brightness_bucket(1.0, CONFIG) == "mid"
    assert brightness_bucket(1.3, CONFIG) == "high"


def test_angle_bucket_left_center_right() -> None:
    assert angle_bucket(-10.0, CONFIG) == "left"
    assert angle_bucket(0.0, CONFIG) == "center"
    assert angle_bucket(10.0, CONFIG) == "right"


def test_background_bucket_dark_light() -> None:
    assert background_bucket((50, 50, 50), CONFIG) == "dark"
    assert background_bucket((230, 230, 230), CONFIG) == "light"


def test_assign_stratum_joins_label_and_buckets() -> None:
    sample = _make_sample(
        brightness=0.7, rotation_deg=-10.0, background=(50, 50, 50), label="defect"
    )

    assert assign_stratum(sample, CONFIG) == "defect_low_left_dark"


def test_stratify_samples_sets_stratum_on_every_sample() -> None:
    samples = [
        _make_sample(0.7, -10.0, (50, 50, 50), "ok"),
        _make_sample(1.3, 10.0, (230, 230, 230), "defect"),
    ]

    stratified = stratify_samples(samples, CONFIG)

    assert stratified[0].stratum == "ok_low_left_dark"
    assert stratified[1].stratum == "defect_high_right_light"
    assert samples[0].stratum is None  # originals untouched


def test_stratum_distribution_counts_and_orders_by_frequency() -> None:
    samples = stratify_samples(
        [
            _make_sample(0.7, -10.0, (50, 50, 50), "ok"),
            _make_sample(0.7, -10.0, (50, 50, 50), "ok"),
            _make_sample(1.3, 10.0, (230, 230, 230), "defect"),
        ],
        CONFIG,
    )

    distribution = stratum_distribution(samples)

    assert distribution == {"ok_low_left_dark": 2, "defect_high_right_light": 1}


def test_label_distribution_zero_fills_unobserved_label() -> None:
    samples = [_make_sample(1.0, 0.0, (200, 200, 200), "ok") for _ in range(3)]

    assert label_distribution(samples) == {"ok": 3, "defect": 0}


def test_brightness_distribution_zero_fills_unobserved_bucket() -> None:
    samples = [_make_sample(0.7, 0.0, (200, 200, 200)) for _ in range(2)]

    assert brightness_distribution(samples, CONFIG) == {"low": 2, "mid": 0, "high": 0}


def test_angle_distribution_counts_all_three_buckets() -> None:
    samples = [
        _make_sample(1.0, -10.0, (200, 200, 200)),
        _make_sample(1.0, 0.0, (200, 200, 200)),
        _make_sample(1.0, 10.0, (200, 200, 200)),
    ]

    assert angle_distribution(samples, CONFIG) == {"left": 1, "center": 1, "right": 1}


def test_background_distribution_counts_both_buckets() -> None:
    samples = [
        _make_sample(1.0, 0.0, (50, 50, 50)),
        _make_sample(1.0, 0.0, (230, 230, 230)),
    ]

    assert background_distribution(samples, CONFIG) == {"dark": 1, "light": 1}


def test_balance_ratio_perfectly_balanced_is_one() -> None:
    assert balance_ratio({"a": 5, "b": 5, "c": 5}) == 1.0


def test_balance_ratio_skewed_distribution() -> None:
    assert balance_ratio({"a": 10, "b": 2}) == pytest.approx(0.2)


def test_balance_ratio_unobserved_bucket_is_zero() -> None:
    assert balance_ratio({"a": 10, "b": 0}) == 0.0


def test_balance_ratio_empty_distribution_is_one() -> None:
    assert balance_ratio({}) == 1.0


def test_compute_distribution_report_balanced_dataset_has_no_warnings() -> None:
    # 6 samples, each axis balanced independently: 2 per brightness bucket,
    # 2 per angle bucket, 3 per background bucket, 3 per label.
    dark, light = (50, 50, 50), (230, 230, 230)
    samples = stratify_samples(
        [
            _make_sample(0.7, -10.0, dark, "ok"),  # low, left, dark
            _make_sample(0.7, 0.0, light, "defect"),  # low, center, light
            _make_sample(1.0, 0.0, dark, "ok"),  # mid, center, dark
            _make_sample(1.0, 10.0, light, "defect"),  # mid, right, light
            _make_sample(1.3, 10.0, dark, "ok"),  # high, right, dark
            _make_sample(1.3, -10.0, light, "defect"),  # high, left, light
        ],
        CONFIG,
    )

    report = compute_distribution_report(samples, CONFIG)

    assert report.total_samples == 6
    assert report.label_distribution == {"ok": 3, "defect": 3}
    assert report.brightness_distribution == {"low": 2, "mid": 2, "high": 2}
    assert all(ratio == 1.0 for ratio in report.balance_ratios.values())
    assert report.warnings == ()


def test_compute_distribution_report_flags_imbalanced_axis() -> None:
    # every sample "low" brightness -> mid/high buckets are zero -> ratio 0.0 < 0.3.
    samples = stratify_samples(
        [_make_sample(0.7, 0.0, (200, 200, 200), "ok") for _ in range(5)], CONFIG
    )

    report = compute_distribution_report(samples, CONFIG)

    assert report.balance_ratios["brightness"] == 0.0
    assert any("brightness" in warning for warning in report.warnings)


def test_compute_distribution_report_to_dict_is_json_serializable() -> None:
    samples = stratify_samples([_make_sample(1.0, 0.0, (200, 200, 200), "ok")], CONFIG)

    report = compute_distribution_report(samples, CONFIG)
    payload = report.to_dict()

    json.dumps(payload)  # raises if anything isn't JSON-serializable
    assert payload["total_samples"] == 1
