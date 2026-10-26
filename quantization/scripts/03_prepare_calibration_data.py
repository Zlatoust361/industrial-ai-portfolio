#!/usr/bin/env python3
"""Step 3/8: sample + preprocess a representative image set for INT4/INT8 calibration.

`04_quantize_model.py`'s TensorRT calibrator feeds these images through the
network to compute per-layer activation ranges -- calibration quality
depends on this set actually representing the deployment distribution, so
it's sampled from week 06's real, stratified val subset by default
(`calibration.source_dir`), not a random dev-set grab or synthetic images.

Deterministic: `--seed` (defaults to `model.seed`) drives the sample, so
re-running this step with the same config/source directory picks the same
images -- a different calibration set silently changing the calibration
cache between runs would make `04`/`05`'s output impossible to reproduce.

Not gated to the Jetson: this is plain file I/O + PIL resizing, portable
anywhere -- only the TensorRT-specific steps from `04_quantize_model.py`
onward need the target device.

Input: `<calibration.source_dir>/**/*.{jpg,jpeg,png}`.
Output: `<paths.calibration_data_dir>/*.png` + `manifest.json`.

Usage:
    python3 scripts/03_prepare_calibration_data.py [--config config/config.yaml]
        [--source-dir PATH] [--num-samples N] [--seed N]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_QUANTIZATION_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_QUANTIZATION_ROOT))

from utils.config import load_config  # noqa: E402

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def discover_images(source_dir: Path) -> list[Path]:
    """Find every candidate calibration image under `source_dir`, sorted for determinism.

    Args:
        source_dir: Directory to search recursively.

    Returns:
        Sorted list of image paths (extensions from `IMAGE_EXTENSIONS`, case-insensitive).

    Raises:
        FileNotFoundError: If `source_dir` doesn't exist.
    """
    if not source_dir.is_dir():
        raise FileNotFoundError(
            f"Calibration source directory not found: {source_dir}. Point --source-dir at a "
            "real image directory -- the default (calibration.source_dir in config.yaml) "
            "expects week 06's dataset_pipeline output; run `cd dataset_pipeline && make "
            "pipeline` first if it doesn't exist yet (see dataset_pipeline/README.md)."
        )
    return sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def sample_calibration_images(
    candidates: list[Path], num_samples: int, seed: int
) -> list[Path]:
    """Deterministically sample up to `num_samples` images from `candidates`.

    Args:
        candidates: Every discovered image path (see `discover_images`).
        num_samples: How many to sample. Capped to `len(candidates)` if
            larger -- a warning, not an error (see `main`'s caller).
        seed: RNG seed for reproducibility.

    Returns:
        A sample of `candidates`, in an order fixed by `seed` (not just the
        first N -- that would bias toward whatever `rglob`/sort order surfaces first).

    Raises:
        ValueError: If `candidates` is empty.
    """
    if not candidates:
        raise ValueError("No candidate calibration images found.")
    rng = random.Random(seed)
    count = min(num_samples, len(candidates))
    return rng.sample(candidates, count)


def preprocess_and_save(
    image_paths: list[Path], output_dir: Path, image_size: int
) -> list[dict[str, object]]:
    """Resize each sampled image to a square and save it under `output_dir`.

    Args:
        image_paths: Sampled source image paths (see `sample_calibration_images`).
        output_dir: Destination directory; created if missing.
        image_size: Square edge (pixels) each image is resized to.

    Returns:
        One manifest entry per saved image: `{"source", "output", "width", "height"}`.
    """
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    for index, source_path in enumerate(image_paths):
        image = Image.open(source_path).convert("RGB").resize((image_size, image_size))
        output_path = output_dir / f"calib_{index:04d}.png"
        image.save(output_path)
        manifest.append(
            {
                "source": str(source_path),
                "output": str(output_path),
                "width": image_size,
                "height": image_size,
            }
        )
    return manifest


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `03_prepare_calibration_data.py`."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--source-dir", default=None, help="Overrides calibration.source_dir from config."
    )
    parser.add_argument(
        "--num-samples", type=int, default=None, help="Overrides calibration.num_samples."
    )
    parser.add_argument("--seed", type=int, default=None, help="Overrides model.seed.")
    return parser.parse_args()


def main() -> None:
    """Sample, resize, and save the calibration image set + its manifest."""
    args = parse_args()
    config = load_config(args.config)

    source_dir = Path(args.source_dir) if args.source_dir else config.calibration.source_dir
    num_samples = (
        args.num_samples if args.num_samples is not None else config.calibration.num_samples
    )
    seed = args.seed if args.seed is not None else config.model.seed

    candidates = discover_images(source_dir)
    if num_samples > len(candidates):
        print(
            f"Warning: requested {num_samples} calibration samples but only "
            f"{len(candidates)} candidate images found under '{source_dir}' -- using all of them.",
            file=sys.stderr,
        )

    sampled = sample_calibration_images(candidates, num_samples, seed)
    manifest = preprocess_and_save(
        sampled, config.paths.calibration_data_dir, config.calibration.image_size
    )

    manifest_path = config.paths.calibration_data_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "source_dir": str(source_dir),
                "num_candidates": len(candidates),
                "num_samples": len(manifest),
                "seed": seed,
                "image_size": config.calibration.image_size,
                "images": manifest,
            },
            indent=2,
        )
    )
    print(
        f"Prepared {len(manifest)} calibration images under "
        f"'{config.paths.calibration_data_dir}' (manifest: '{manifest_path}')."
    )


if __name__ == "__main__":
    main()
