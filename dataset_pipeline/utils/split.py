"""Stratified train/val/test split + physically reorganizing stage 4/5's
output to match (week 06, stage 6).

Split is assigned per stratum (largest-remainder rounding keeps each
split's stratum mix close to the whole dataset's), and every augmented
sample (stage 3) inherits its source sample's split -- otherwise a flipped
twin of a train image could land in val/test, leaking the exact same
underlying scene across the split boundary.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from utils.config import SplitConfig
from utils.manifest import Sample, with_split

SPLITS: tuple[str, ...] = ("train", "val", "test")


def assign_splits(samples: list[Sample], config: SplitConfig) -> list[Sample]:
    """Assign `train`/`val`/`test` to every sample, stratified by `stratum`.

    Args:
        samples: Samples with `stratum` already set (stage 2); may include
            augmented samples (stage 3, `source_sample_id` set).
        config: Split fractions (must already sum to ~1.0, enforced by
            `utils.config.load_config`) and shuffle seed.

    Returns:
        Copies of `samples`, in the same order, with `split` set.

    Raises:
        ValueError: If an augmented sample's `source_sample_id` doesn't
            match any base sample in `samples`.
    """
    base_samples = [s for s in samples if s.source_sample_id is None]
    augmented_samples = [s for s in samples if s.source_sample_id is not None]

    by_stratum: dict[str, list[Sample]] = {}
    for sample in base_samples:
        by_stratum.setdefault(sample.stratum, []).append(sample)

    rng = np.random.default_rng(config.seed)
    split_by_sample_id: dict[str, str] = {}

    fractions = {
        "train": config.train_fraction,
        "val": config.val_fraction,
        "test": config.test_fraction,
    }

    for stratum in sorted(by_stratum):
        group = by_stratum[stratum]
        order = rng.permutation(len(group))
        n = len(group)

        # Largest-remainder (Hamilton) apportionment: floor each split's raw
        # share, then hand out the few leftover slots to whichever split has
        # the largest fractional remainder. A naive "dump the rounding
        # leftover into test" would starve train on every single/double-
        # sample stratum (common here: a 60-sample dataset spread over up to
        # dozens of label x brightness x angle x background combinations) --
        # e.g. a 1-sample stratum would ALWAYS land in test, biasing the
        # aggregate split far off the configured fractions.
        raw_shares = {split: n * fraction for split, fraction in fractions.items()}
        counts = {split: int(raw_shares[split]) for split in SPLITS}
        remaining = n - sum(counts.values())
        by_largest_remainder = sorted(
            SPLITS, key=lambda split: raw_shares[split] - counts[split], reverse=True
        )
        for split in by_largest_remainder[:remaining]:
            counts[split] += 1

        rank_to_split = (
            ["train"] * counts["train"] + ["val"] * counts["val"] + ["test"] * counts["test"]
        )
        for rank, sample_index in enumerate(order):
            split_by_sample_id[group[sample_index].sample_id] = rank_to_split[rank]

    for sample in augmented_samples:
        source_split = split_by_sample_id.get(sample.source_sample_id)
        if source_split is None:
            raise ValueError(
                f"Augmented sample {sample.sample_id!r} references unknown "
                f"source_sample_id {sample.source_sample_id!r}."
            )
        split_by_sample_id[sample.sample_id] = source_split

    return [with_split(sample, split_by_sample_id[sample.sample_id]) for sample in samples]


def split_distribution(samples: list[Sample]) -> dict[str, dict[str, int]]:
    """Count samples per `(split, stratum)`, for a quick balance sanity check.

    Args:
        samples: Samples with `split` already assigned.

    Returns:
        `{split: {stratum: count}}`.
    """
    distribution: dict[str, dict[str, int]] = {split: {} for split in SPLITS}
    for sample in samples:
        bucket = distribution.setdefault(sample.split, {})
        bucket[sample.stratum] = bucket.get(sample.stratum, 0) + 1
    return distribution


def reorganize_yolo_by_split(yolo_dir: str | Path, samples: list[Sample]) -> None:
    """Move stage 5's flat `images/`/`labels/` into per-split subdirectories.

    Each split subdirectory (`yolo_dir/{train,val,test}/`) ends up a
    complete, independent YOLO root -- `images/`, `labels/`, its own copy
    of `classes.txt` -- directly loadable by
    `src.evaluation.detection_metrics.load_yolo_test_set`.

    Args:
        yolo_dir: Stage 5's YOLO output root (has `images/`, `labels/`, `classes.txt`).
        samples: Samples with `split` assigned; matched to files by filename stem.

    Raises:
        FileNotFoundError: If a sample's image/label file is missing.
    """
    yolo_dir = Path(yolo_dir)
    images_dir = yolo_dir / "images"
    labels_dir = yolo_dir / "labels"
    classes_path = yolo_dir / "classes.txt"

    for split in SPLITS:
        (yolo_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (yolo_dir / split / "labels").mkdir(parents=True, exist_ok=True)
        shutil.copy2(classes_path, yolo_dir / split / "classes.txt")

    for sample in samples:
        filename = Path(sample.image_path).name
        stem = Path(filename).stem
        src_image = images_dir / filename
        src_label = labels_dir / f"{stem}.txt"
        if not src_image.is_file() or not src_label.is_file():
            raise FileNotFoundError(f"Missing YOLO image/label for sample {sample.sample_id!r}.")
        shutil.move(str(src_image), str(yolo_dir / sample.split / "images" / filename))
        shutil.move(str(src_label), str(yolo_dir / sample.split / "labels" / f"{stem}.txt"))

    images_dir.rmdir()
    labels_dir.rmdir()


def partition_coco_by_split(
    coco_path: str | Path, samples: list[Sample], output_dir: str | Path
) -> None:
    """Split stage 4's single COCO JSON into one file per split.

    Args:
        coco_path: Stage 4's `build_coco_dataset` output file.
        samples: Samples with `split` assigned; matched to COCO images by filename.
        output_dir: Destination directory for `annotations_{train,val,test}.json`.

    Raises:
        FileNotFoundError: If `coco_path` doesn't exist.
    """
    coco_path = Path(coco_path)
    if not coco_path.is_file():
        raise FileNotFoundError(f"COCO annotations file not found: {coco_path}")
    dataset = json.loads(coco_path.read_text(encoding="utf-8"))

    split_by_filename = {Path(s.image_path).name: s.split for s in samples}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        keep_ids = {
            image["id"]
            for image in dataset["images"]
            if split_by_filename.get(Path(image["file_name"]).name) == split
        }
        split_dataset = {
            "images": [img for img in dataset["images"] if img["id"] in keep_ids],
            "annotations": [ann for ann in dataset["annotations"] if ann["image_id"] in keep_ids],
            "categories": dataset["categories"],
        }
        (output_dir / f"annotations_{split}.json").write_text(json.dumps(split_dataset, indent=2))
