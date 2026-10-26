#!/usr/bin/env python3
"""Stage 6/7: stratified train/val/test split (week 06).

Assigns `train`/`val`/`test` per sample (`utils.split.assign_splits`,
stratified by stage 2's `stratum`, augmented samples inheriting their
source's split), then physically reorganizes stage 4/5's output to match:
the flat YOLO `images/`/`labels/` become `{train,val,test}/images/` +
`{train,val,test}/labels/`, and the single COCO file is partitioned into
`annotations_{train,val,test}.json`. Must run after stages 4 and 5 --
"convert to COCO/YOLO -> split" is the pipeline's own stated order (see
`course/weeks/week06-synthetic-dataset-pipeline/lab.md`), not "split ->
convert", so both output formats stay consistent with the same split.

Input: `<data_dir>/manifest_augmented.json` (stage 3), `<outputs_dir>/coco/annotations.json`
(stage 4), `<outputs_dir>/yolo/` (stage 5).
Output: `<data_dir>/manifest_split.json`, `<outputs_dir>/yolo/{train,val,test}/`,
`<outputs_dir>/coco/annotations_{train,val,test}.json`.

Usage:
    python3 scripts/06_split_dataset.py [--config config/dataset_config.yaml]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import load_config  # noqa: E402
from utils.manifest import load_manifest, save_manifest  # noqa: E402
from utils.split import (  # noqa: E402
    SPLITS,
    assign_splits,
    partition_coco_by_split,
    reorganize_yolo_by_split,
    split_distribution,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `06_split_dataset.py`."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config/dataset_config.yaml")
    return parser.parse_args()


def main() -> None:
    """Assign splits, reorganize the YOLO/COCO output to match, write `manifest_split.json`."""
    args = parse_args()
    config = load_config(args.config)

    input_path = config.paths.data_dir / "manifest_augmented.json"
    try:
        samples = load_manifest(input_path)
    except FileNotFoundError as exc:
        print(f"Error: {exc} (run 03_augment_dataset.py first).", file=sys.stderr)
        sys.exit(1)

    split_samples = assign_splits(samples, config.split)

    yolo_dir = config.paths.outputs_dir / "yolo"
    try:
        reorganize_yolo_by_split(yolo_dir, split_samples)
    except FileNotFoundError as exc:
        print(f"Error: {exc} (run 05_convert_yolo.py first).", file=sys.stderr)
        sys.exit(1)

    coco_path = config.paths.outputs_dir / "coco" / "annotations.json"
    try:
        partition_coco_by_split(coco_path, split_samples, config.paths.outputs_dir / "coco")
    except FileNotFoundError as exc:
        print(f"Error: {exc} (run 04_convert_coco.py first).", file=sys.stderr)
        sys.exit(1)

    manifest_path = config.paths.data_dir / "manifest_split.json"
    save_manifest(manifest_path, split_samples)

    distribution = split_distribution(split_samples)
    print("Split sizes:")
    for split in SPLITS:
        num_samples = sum(distribution[split].values())
        num_strata = len(distribution[split])
        print(f"  {split}: {num_samples} sample(s), {num_strata} strata")
    print(f"YOLO dataset reorganized under '{yolo_dir}/{{train,val,test}}/'.")
    print(f"COCO annotations partitioned under '{config.paths.outputs_dir / 'coco'}/'.")
    print(f"Manifest written to '{manifest_path}'.")


if __name__ == "__main__":
    main()
