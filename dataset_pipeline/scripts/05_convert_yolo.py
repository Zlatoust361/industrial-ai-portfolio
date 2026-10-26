#!/usr/bin/env python3
"""Stage 5/7: convert the manifest to a YOLO-format dataset (week 06).

Same `class_id cx cy w h` normalized-line convention as
`src.evaluation.detection_metrics` (week 07) -- see `utils.yolo`'s module
docstring for why this matters (this output is directly consumable by
week 07's training/comparison scripts). Day 3 adds a dataset-integrity
validator alongside this and stage 4's COCO writer.

Input: `<data_dir>/manifest_augmented.json` (stage 3).
Output: `<outputs_dir>/yolo/{images,labels}/*`, `<outputs_dir>/yolo/classes.txt`.

Usage:
    python3 scripts/05_convert_yolo.py [--config config/dataset_config.yaml]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import load_config  # noqa: E402
from utils.manifest import load_manifest  # noqa: E402
from utils.yolo import write_yolo_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `05_convert_yolo.py`."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config/dataset_config.yaml")
    return parser.parse_args()


def main() -> None:
    """Write `outputs_dir/yolo/` from the stage-3 manifest."""
    args = parse_args()
    config = load_config(args.config)

    input_path = config.paths.data_dir / "manifest_augmented.json"
    try:
        samples = load_manifest(input_path)
    except FileNotFoundError as exc:
        print(f"Error: {exc} (run 03_augment_dataset.py first).", file=sys.stderr)
        sys.exit(1)

    output_dir = config.paths.outputs_dir / "yolo"
    write_yolo_dataset(
        samples,
        config.paths.data_dir,
        config.generation.object_class_name,
        config.generation.defect_class_name,
        output_dir,
    )

    print(f"Wrote YOLO dataset for {len(samples)} sample(s) to '{output_dir}'.")


if __name__ == "__main__":
    main()
