#!/usr/bin/env python3
"""Stage 7/7: export a compact, stratified val subset for Jetson Orin Nano profiling (week 06).

Picks `jetson_subset.target_count` images (200-300 by default) from the
val split, proportionally across strata (`utils.jetson_subset`), so the
subset stays representative rather than e.g. all-bright or all-`ok`. The
full dataset is never copied to the device -- limited NVMe/microSD
on-device storage is exactly the problem this stage solves; profiling
latency/mAP needs a stable, small, representative sample, not every image.

Input: `<data_dir>/manifest_split.json` (stage 6), `<outputs_dir>/yolo/val/` (stage 6).
Output: `<outputs_dir>/jetson_val_subset/{images,labels}/*`,
`<outputs_dir>/jetson_val_subset/classes.txt`.

Usage:
    python3 scripts/07_export_jetson_val_subset.py [--config config/dataset_config.yaml]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import load_config  # noqa: E402
from utils.jetson_subset import export_jetson_subset, select_stratified_subset  # noqa: E402
from utils.manifest import load_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `07_export_jetson_val_subset.py`."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config/dataset_config.yaml")
    return parser.parse_args()


def main() -> None:
    """Select + export a stratified Jetson-sized val subset."""
    args = parse_args()
    config = load_config(args.config)

    manifest_path = config.paths.data_dir / "manifest_split.json"
    try:
        samples = load_manifest(manifest_path)
    except FileNotFoundError as exc:
        print(f"Error: {exc} (run 06_split_dataset.py first).", file=sys.stderr)
        sys.exit(1)

    val_samples = [sample for sample in samples if sample.split == "val"]

    try:
        subset = select_stratified_subset(val_samples, config.jetson_subset)
    except ValueError as exc:
        print(
            f"Error: {exc} Increase generation.count in dataset_config.yaml "
            "(a small demo run doesn't have enough val samples yet) or lower "
            "jetson_subset.min_count.",
            file=sys.stderr,
        )
        sys.exit(1)

    val_yolo_dir = config.paths.outputs_dir / "yolo" / "val"
    output_dir = config.paths.outputs_dir / "jetson_val_subset"
    try:
        output_dir, total_bytes = export_jetson_subset(subset, val_yolo_dir, output_dir)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    total_mb = total_bytes / (1024 * 1024)
    print(f"Exported {len(subset)} image(s) from {len(val_samples)} val sample(s) available.")
    print(f"Total size: {total_mb:.1f} MB.")
    print(f"Jetson val subset written to '{output_dir}'.")


if __name__ == "__main__":
    main()
