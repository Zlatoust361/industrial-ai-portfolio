"""Paired statistical significance testing (week 07, day 4).

`paired_t_test` wraps `scipy.stats.ttest_rel`, lazily imported so this
module (and everything that imports it, like
`scripts/compare_dr_branches.py`) still loads cleanly if `scipy` isn't
installed -- consistent with how `torch`/`transformers`/`ultralytics` are
handled elsewhere in this repo. `scipy` genuinely is a new dependency
(added to `requirements.txt`) rather than a hand-rolled t-distribution CDF:
the numerics of an accurate incomplete-beta-function-based p-value are
easy to get subtly wrong, and this is exactly the kind of well-tested,
narrow piece of math a real statistics library should own.
"""

from __future__ import annotations


def paired_t_test(sample_a: list[float], sample_b: list[float]) -> dict[str, float]:
    """Paired (dependent-samples) two-sided t-test between two equal-length samples.

    "Paired" here means `sample_a[i]`/`sample_b[i]` are the same underlying
    unit measured under two conditions -- e.g. the same test image's F1
    score under the `no_dr`-trained model vs the `with_dr`-trained model
    (see `src.evaluation.detection_metrics.per_image_f1_scores`). This is
    the correct test for that setup: an unpaired test would ignore that
    the two samples are correlated (same images), and either overstate or
    understate significance depending on how they're correlated.

    Args:
        sample_a: First condition's per-unit measurements.
        sample_b: Second condition's per-unit measurements, same units,
            same order, same length as `sample_a`.

    Returns:
        `{"t_statistic", "p_value", "mean_difference", "n"}`.
        `mean_difference` is `mean(sample_b) - mean(sample_a)` (positive
        means `sample_b` scored higher on average).

    Raises:
        ValueError: If `sample_a`/`sample_b` have different lengths, or
            fewer than 2 paired observations (a t-test needs at least 2 to
            estimate variance at all).
    """
    if len(sample_a) != len(sample_b):
        raise ValueError(
            f"paired_t_test needs equal-length samples, got {len(sample_a)} vs {len(sample_b)}."
        )
    if len(sample_a) < 2:
        raise ValueError(
            f"paired_t_test needs at least 2 paired observations, got {len(sample_a)}."
        )

    from scipy import stats

    result = stats.ttest_rel(sample_b, sample_a)
    mean_difference = sum(sample_b) / len(sample_b) - sum(sample_a) / len(sample_a)

    return {
        "t_statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "mean_difference": mean_difference,
        "n": len(sample_a),
    }
