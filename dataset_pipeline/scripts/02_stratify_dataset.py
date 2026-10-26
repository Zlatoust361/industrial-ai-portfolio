#!/usr/bin/env python3
"""Stage 2/8: assign a stratum + log balance by brightness/angle/background (week 06, day 4).

Discretizes each sample's *actual* DR render parameters
(`Sample.dr_params`, stage 1's output) into brightness/angle/background
buckets (`utils.stratify`), joined with its `ok`/`defect` label into one
`stratum` string -- stages 6/7 both stratified-sample using this field.

Also computes and logs the dataset's balance: each axis's *marginal*
distribution (brightness alone, angle alone, background alone, label
alone) plus the joint `stratum` distribution, and a `[0, 1]` balance
ratio per axis (smallest bucket's share of the largest -- `1.0` =
perfectly even, `0.0` = a bucket was never observed at all). An axis
below `stratify.balance_warn_ratio` gets a warning on stderr and in the
report -- informational, not a hard gate (that's `08_validate_dataset.py`'s
job, scoped to correctness, not balance).

Input: `<data_dir>/manifest_generated.json` (stage 1).
Output: `<data_dir>/manifest_stratified.json`, `<outputs_dir>/metrics/stratify_distribution.json`.

Usage:
    python3 scripts/02_stratify_dataset.py [--config config/dataset_config.yaml]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import load_config  # noqa: E402
from utils.manifest import load_manifest, save_manifest  # noqa: E402
from utils.stratify import (  # noqa: E402
    DistributionReport,
    compute_distribution_report,
    stratify_samples,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `02_stratify_dataset.py`."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config/dataset_config.yaml")
    return parser.parse_args()


def print_report(report: DistributionReport) -> None:
    """Print a human-readable summary of `report` to stdout, warnings to stderr.

    Args:
        report: The computed distribution/balance report.
    """
    num_strata = len(report.stratum_distribution)
    print(f"Stratified {report.total_samples} sample(s) into {num_strata} strata.")
    print("Marginal distributions (balance ratio -- 1.0 = perfectly even, 0.0 = empty bucket):")
    for axis, distribution in (
        ("label", report.label_distribution),
        ("brightness", report.brightness_distribution),
        ("angle", report.angle_distribution),
        ("background", report.background_distribution),
    ):
        print(f"  {axis} (ratio {report.balance_ratios[axis]:.2f}): {distribution}")
    print("Joint strata:")
    for stratum, count in report.stratum_distribution.items():
        print(f"  {stratum}: {count}")

    for warning in report.warnings:
        print(f"Warning: {warning}", file=sys.stderr)


def main() -> None:
    """Assign strata, compute the balance report, write the manifest + metrics report."""
    args = parse_args()
    config = load_config(args.config)

    input_path = config.paths.data_dir / "manifest_generated.json"
    try:
        samples = load_manifest(input_path)
    except FileNotFoundError as exc:
        print(f"Error: {exc} (run 01_generate_synthetic.py first).", file=sys.stderr)
        sys.exit(1)

    stratified = stratify_samples(samples, config.stratify)

    manifest_path = config.paths.data_dir / "manifest_stratified.json"
    save_manifest(manifest_path, stratified)

    report = compute_distribution_report(stratified, config.stratify)
    print_report(report)

    metrics_path = config.paths.outputs_dir / "metrics" / "stratify_distribution.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(report.to_dict(), indent=2))

    print(f"Manifest written to '{manifest_path}'.")
    print(f"Distribution report written to '{metrics_path}'.")


if __name__ == "__main__":
    main()
