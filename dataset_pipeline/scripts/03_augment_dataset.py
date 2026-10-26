#!/usr/bin/env python3
"""Stage 3/7: add an augmented variant for a fraction of samples (week 06).

Picks `augment.augment_fraction` of stage-2 samples, applies one randomly
chosen augmentation each (`utils.augment.AUGMENTATIONS`), and appends the
result as a new sample -- `source_sample_id`/`augmentation` set,
`stratum` inherited from the source (an augmented image's condition
bucket doesn't change; hflip/brightness/contrast/noise don't move it into
a different lighting/angle/background regime). Stage 6's split assigns
splits per `source_sample_id`, not per augmented sample, so a flipped
twin can never leak across the train/val/test boundary.

Input: `<data_dir>/manifest_stratified.json` (stage 2).
Output: `<data_dir>/images/*_aug_*.jpg` (new files), `<data_dir>/manifest_augmented.json`.

Usage:
    python3 scripts/03_augment_dataset.py [--config config/dataset_config.yaml]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.augment import AUGMENTATIONS, apply_augmentation  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.manifest import Sample, load_manifest, save_manifest  # noqa: E402


def augment_dataset(
    samples: list[Sample], data_dir: Path, augment_fraction: float, rng: np.random.Generator
) -> list[Sample]:
    """Generate one augmented variant each for a random `augment_fraction` of `samples`.

    Args:
        samples: Stage-2 samples (base samples only -- this stage is
            idempotent-unsafe to re-run on its own output, since every
            sample would then be eligible for augmentation again).
        data_dir: Where each source/augmented image lives/gets written.
        augment_fraction: Fraction of `samples` to augment, `[0, 1]`.
        rng: Source of randomness (which samples, which augmentation, jitter strength).

    Returns:
        `samples` plus one new `Sample` per augmented source, each with
        `source_sample_id`/`augmentation` set and the source's `stratum`.
    """
    num_to_augment = round(len(samples) * augment_fraction)
    chosen_indices = rng.choice(len(samples), size=num_to_augment, replace=False)

    augmented: list[Sample] = []
    for index in sorted(chosen_indices):
        source = samples[index]
        augmentation = str(rng.choice(AUGMENTATIONS))

        source_image = Image.open(data_dir / source.image_path).convert("RGB")
        new_image, object_bbox, defect_boxes = apply_augmentation(
            source_image, source.object_bbox, list(source.defect_boxes), augmentation, rng
        )

        new_sample_id = f"{source.sample_id}_aug_{augmentation}"
        new_image_path = f"images/{new_sample_id}.jpg"
        new_image.save(data_dir / new_image_path, quality=95)

        augmented.append(
            Sample(
                sample_id=new_sample_id,
                image_path=new_image_path,
                width=new_image.width,
                height=new_image.height,
                label=source.label,
                object_bbox=object_bbox,
                defect_boxes=tuple(defect_boxes),
                dr_params=source.dr_params,
                stratum=source.stratum,
                source_sample_id=source.sample_id,
                augmentation=augmentation,
            )
        )

    return [*samples, *augmented]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `03_augment_dataset.py`."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config/dataset_config.yaml")
    return parser.parse_args()


def main() -> None:
    """Augment a fraction of stage-2 samples and write `manifest_augmented.json`."""
    args = parse_args()
    config = load_config(args.config)

    input_path = config.paths.data_dir / "manifest_stratified.json"
    try:
        samples = load_manifest(input_path)
    except FileNotFoundError as exc:
        print(f"Error: {exc} (run 02_stratify_dataset.py first).", file=sys.stderr)
        sys.exit(1)

    rng = np.random.default_rng(config.augment.seed)
    augmented = augment_dataset(
        samples, config.paths.data_dir, config.augment.augment_fraction, rng
    )

    output_path = config.paths.data_dir / "manifest_augmented.json"
    save_manifest(output_path, augmented)

    num_added = len(augmented) - len(samples)
    print(f"Added {num_added} augmented sample(s) to {len(samples)} base sample(s).")
    print(f"Manifest written to '{output_path}'.")


if __name__ == "__main__":
    main()
