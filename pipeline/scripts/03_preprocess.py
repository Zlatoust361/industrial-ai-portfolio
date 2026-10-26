#!/usr/bin/env python3
"""Stage 3/7: letterbox-resize ingested images, write `data/processed/` + manifest.

Resizes every image from `data/manifest.json` to `preprocess.image_size`,
preserving aspect ratio and centering it on a fixed-size canvas (see
`utils.preprocess.compute_letterbox_geometry`) -- a naive per-axis stretch
would distort defects, which is exactly the signal stage 4 needs to detect.
This resize is for a consistent, inspectable `data/processed/` artifact;
`04_detect.py`'s CLIP backend still runs its own processor over the
original-resolution image, since that's the input distribution the model
was actually trained on.

Input: `data/manifest.json` (from `02_ingest.py`).
Output: `data/processed/*` (resized images) + `data/processed_manifest.json`.

Usage:
    python3 scripts/03_preprocess.py [--config config/pipeline_config.yaml]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import load_config  # noqa: E402
from utils.preprocess import compute_letterbox_geometry  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `03_preprocess.py`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/pipeline_config.yaml")
    return parser.parse_args()


def main() -> None:
    """Letterbox-resize every manifested image into `data/processed/`."""
    from PIL import Image

    args = parse_args()
    config = load_config(args.config)

    manifest_path = config.paths.data_dir / "manifest.json"
    if not manifest_path.is_file():
        print(
            f"Error: manifest not found at '{manifest_path}' -- run 02_ingest.py first.",
            file=sys.stderr,
        )
        sys.exit(1)
    manifest = json.loads(manifest_path.read_text())

    processed_dir = config.paths.data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    target_size = config.preprocess.image_size

    processed_records = []
    for entry in manifest["images"]:
        source_path = Path(entry["path"])
        geometry = compute_letterbox_geometry(
            original_size=(entry["width"], entry["height"]), target_size=target_size
        )

        with Image.open(source_path) as image:
            resized = image.convert("RGB").resize(geometry.resized_size, Image.LANCZOS)
            canvas = Image.new("RGB", target_size, color=(114, 114, 114))
            canvas.paste(resized, geometry.offset)

        processed_path = processed_dir / source_path.name
        canvas.save(processed_path)

        processed_records.append(
            {
                "source_path": str(source_path),
                "processed_path": str(processed_path),
                "original_size": [entry["width"], entry["height"]],
                "target_size": list(target_size),
                "scale": geometry.scale,
                "offset": list(geometry.offset),
            }
        )

    processed_manifest_path = config.paths.data_dir / "processed_manifest.json"
    processed_manifest_path.write_text(
        json.dumps({"target_size": list(target_size), "images": processed_records}, indent=2)
    )

    print(f"Preprocessed {len(processed_records)} image(s) to {target_size}.")
    print(f"Processed manifest written to '{processed_manifest_path}'.")


if __name__ == "__main__":
    main()
