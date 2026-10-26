"""CLI: batch-generate parametric defect masks (week 05, day 2).

Wraps `src.synthetic.mask_generation.generate_defect_mask` over either a
folder of real base images (one mask per image, sized to match) or a
`--count` of synthetic blank canvases (when no base images are available --
e.g. exercising this script in this sandbox, no client data involved).
Writes each mask as a PNG plus one `manifest.json` recording exactly what
was drawn (shape/type/bbox per instance, coverage fraction) -- day 3's
`diffusion_pipeline` consumes both the masks and this manifest.

Usage:
    PYTHONPATH=. python scripts/generate_defect_masks.py \\
        --image-dir path/to/base_images [--config config/defects.yaml] \\
        [--output-dir outputs/synthetic_defects/masks] [--seed 42]

    # No base images on hand -- generate N blank placeholder canvases instead:
    PYTHONPATH=. python scripts/generate_defect_masks.py \\
        --image-size 512x512 --count 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.preprocessing import load_image, parse_image_size
from src.synthetic.config import load_defects_config
from src.synthetic.mask_generation import generate_defect_mask, mask_coverage_fraction

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `generate_defect_masks.py`."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config/defects.yaml")
    parser.add_argument(
        "--image-dir",
        default=None,
        help="Folder of base images to size masks against (one mask per image). "
        "If omitted, --count blank canvases of --image-size are used instead.",
    )
    parser.add_argument(
        "--image-size",
        default="512x512",
        help="WIDTHxHEIGHT for synthetic canvases when --image-dir is not given.",
    )
    parser.add_argument(
        "--count", type=int, default=8, help="Number of masks when --image-dir is not given."
    )
    parser.add_argument("--output-dir", default="outputs/synthetic_defects/masks")
    parser.add_argument("--seed", type=int, default=None, help="Override config.seed.")
    return parser.parse_args()


def _collect_base_image_sizes(image_dir: Path) -> list[tuple[str, tuple[int, int]]]:
    """List `(stem, (width, height))` for every image under `image_dir`, sorted by name.

    Raises:
        FileNotFoundError: If `image_dir` doesn't exist or has no matching images.
    """
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise FileNotFoundError(f"No images ({sorted(IMAGE_EXTENSIONS)}) found under: {image_dir}")
    return [(path.stem, load_image(path).size) for path in paths]


def main() -> None:
    """Generate masks, write each as a PNG, write `manifest.json` alongside them."""
    args = parse_args()
    config = load_defects_config(args.config)
    if args.seed is not None:
        from dataclasses import replace

        config = replace(config, seed=args.seed)
    rng = np.random.default_rng(config.seed)

    if args.image_dir is not None:
        try:
            targets = _collect_base_image_sizes(Path(args.image_dir))
        except FileNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        image_size = parse_image_size(args.image_size)
        targets = [(f"mask_{i:03d}", image_size) for i in range(args.count)]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries = []
    for name, image_size in targets:
        mask, instances = generate_defect_mask(image_size, config, rng=rng)
        mask_path = output_dir / f"{name}.png"
        Image.fromarray(mask, mode="L").save(mask_path)
        manifest_entries.append(
            {
                "mask": str(mask_path),
                "image_size": list(image_size),
                "coverage_fraction": mask_coverage_fraction(mask),
                "instances": [instance.to_dict() for instance in instances],
            }
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {"config": str(args.config), "seed": config.seed, "masks": manifest_entries}, indent=2
        )
    )

    print(f"Generated {len(manifest_entries)} mask(s) under '{output_dir}'.")
    print(f"Manifest written to '{manifest_path}'.")


if __name__ == "__main__":
    main()
