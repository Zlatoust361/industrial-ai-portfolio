#!/usr/bin/env python3
"""Stage 4/7: convert the manifest to a single COCO-format annotations file (week 06).

Two categories: `generation.object_class_name` (every sample's
`object_bbox`) and `generation.defect_class_name` (each `defect_boxes`
entry). Day 3 adds a dataset-integrity validator alongside this and stage
5's YOLO writer -- this stage's own output is already a real, spec-shaped
COCO file (see `utils.coco`).

Input: `<data_dir>/manifest_augmented.json` (stage 3).
Output: `<outputs_dir>/coco/annotations.json`.

Usage:
    python3 scripts/04_convert_coco.py [--config config/dataset_config.yaml]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.coco import write_coco_dataset  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.manifest import load_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `04_convert_coco.py`."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config/dataset_config.yaml")
    return parser.parse_args()


def main() -> None:
    """Write `outputs_dir/coco/annotations.json` from the stage-3 manifest."""
    args = parse_args()
    config = load_config(args.config)

    input_path = config.paths.data_dir / "manifest_augmented.json"
    try:
        samples = load_manifest(input_path)
    except FileNotFoundError as exc:
        print(f"Error: {exc} (run 03_augment_dataset.py first).", file=sys.stderr)
        sys.exit(1)

    output_path = config.paths.outputs_dir / "coco" / "annotations.json"
    write_coco_dataset(
        samples,
        config.generation.object_class_name,
        config.generation.defect_class_name,
        output_path,
    )

    print(f"Wrote COCO annotations for {len(samples)} sample(s) to '{output_path}'.")


if __name__ == "__main__":
    main()
