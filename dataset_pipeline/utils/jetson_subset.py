"""Stratified val-only subset selection for Jetson Orin Nano profiling (week 06, stage 7).

Picks `target_count` images from the val split, proportionally across
strata (largest-remainder apportionment, same reasoning as
`utils.split.assign_splits`), so the subset stays representative rather
than e.g. all-bright or all-`ok`. The full dataset is never copied to the
device -- limited NVMe/microSD storage is the whole point of this stage.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from utils.config import JetsonSubsetConfig
from utils.manifest import Sample


def select_stratified_subset(val_samples: list[Sample], config: JetsonSubsetConfig) -> list[Sample]:
    """Pick a stratified subset of `val_samples`, sized within `[min_count, max_count]`.

    Args:
        val_samples: Samples with `split == "val"`.
        config: `target_count`/`min_count`/`max_count`, `seed`.

    Returns:
        A stratified subsample, sorted by `(stratum, sample_id)` for
        reproducible, diffable output across re-runs of the same config.

    Raises:
        ValueError: If `val_samples` is empty, or has fewer than `min_count` samples.
    """
    if not val_samples:
        raise ValueError("select_stratified_subset needs a non-empty val_samples.")
    if len(val_samples) < config.min_count:
        raise ValueError(
            f"Only {len(val_samples)} val samples available, need at least "
            f"{config.min_count} (see jetson_subset.min_count)."
        )

    target = min(config.target_count, len(val_samples), config.max_count)

    by_stratum: dict[str, list[Sample]] = {}
    for sample in val_samples:
        by_stratum.setdefault(sample.stratum, []).append(sample)
    strata = sorted(by_stratum)

    raw_shares = {
        stratum: len(by_stratum[stratum]) / len(val_samples) * target for stratum in strata
    }
    counts = {
        stratum: min(int(raw_shares[stratum]), len(by_stratum[stratum])) for stratum in strata
    }
    remaining = target - sum(counts.values())
    by_largest_remainder = sorted(strata, key=lambda s: raw_shares[s] - counts[s], reverse=True)
    for stratum in by_largest_remainder:
        if remaining <= 0:
            break
        if counts[stratum] < len(by_stratum[stratum]):
            counts[stratum] += 1
            remaining -= 1

    rng = np.random.default_rng(config.seed)
    selected: list[Sample] = []
    for stratum in strata:
        group = by_stratum[stratum]
        picked_indices = rng.permutation(len(group))[: counts[stratum]]
        selected.extend(group[i] for i in picked_indices)

    return sorted(selected, key=lambda sample: (sample.stratum, sample.sample_id))


def export_jetson_subset(
    subset: list[Sample], val_yolo_dir: str | Path, output_dir: str | Path
) -> tuple[Path, int]:
    """Copy `subset`'s YOLO images/labels into a compact, standalone dataset.

    Args:
        subset: Samples to export (from `select_stratified_subset`).
        val_yolo_dir: Stage 6's val-split YOLO root (`images/`, `labels/`, `classes.txt`).
        output_dir: Destination root.

    Returns:
        `(output_dir, total_bytes)` -- the on-disk size of everything copied,
        for the "hundreds of MB, not gigabytes" sanity check.

    Raises:
        FileNotFoundError: If a sample's image/label file is missing.
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    classes_dest = output_dir / "classes.txt"
    shutil.copy2(Path(val_yolo_dir) / "classes.txt", classes_dest)
    total_bytes = classes_dest.stat().st_size

    for sample in subset:
        filename = Path(sample.image_path).name
        stem = Path(filename).stem
        src_image = Path(val_yolo_dir) / "images" / filename
        src_label = Path(val_yolo_dir) / "labels" / f"{stem}.txt"
        if not src_image.is_file() or not src_label.is_file():
            raise FileNotFoundError(f"Missing YOLO image/label for sample {sample.sample_id!r}.")
        dest_image = images_dir / filename
        dest_label = labels_dir / f"{stem}.txt"
        shutil.copy2(src_image, dest_image)
        shutil.copy2(src_label, dest_label)
        total_bytes += dest_image.stat().st_size + dest_label.stat().st_size

    return output_dir, total_bytes
