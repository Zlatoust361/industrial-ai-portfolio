#!/usr/bin/env python3
"""Generates a tiny synthetic demo dataset under `<data_dir>/raw/`.

`run_all.sh` calls this automatically when `<data_dir>/raw/` is empty, so a
fresh checkout can run the full 7-stage pipeline with zero manual setup and
no real inspection images. Every image is a plain gray square; a `--defects`
fraction of them additionally get a bright-red marker rectangle painted on
(RGB ~ `(220, 30, 30)`) that `04_detect.py`'s `stub` backend
(`StubDetector`) looks for -- so this generator and that backend together
form a real, if trivial, end-to-end signal: not a hardcoded score.

NOT a substitute for a real dataset in production -- see
`pipeline/README.md`'s "CI/demo backend" section.

Usage:
    python3 scripts/generate_demo_dataset.py [--config config/pipeline_config.yaml]
        [--count 12] [--defects 0]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import load_config  # noqa: E402

MARKER_RGB = (220, 30, 30)
"""Must match `04_detect.py::StubDetector._MARKER_RGB`."""


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `generate_demo_dataset.py`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/pipeline_config.yaml")
    parser.add_argument("--count", type=int, default=12, help="Total demo images to generate.")
    parser.add_argument(
        "--defects",
        type=int,
        default=0,
        help="How many of --count images get the defect marker (default 0 -> a clean "
        "GREEN demo run; pass e.g. 1 to see a YELLOW/RED verdict instead).",
    )
    return parser.parse_args()


def main() -> None:
    """Write `--count` synthetic images (`--defects` of them marked) to `<data_dir>/raw/`."""
    from PIL import Image, ImageDraw

    args = parse_args()
    if not 0 <= args.defects <= args.count:
        print("Error: --defects must be between 0 and --count.", file=sys.stderr)
        sys.exit(1)
    config = load_config(args.config)

    raw_dir = config.paths.data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.count):
        shade = 120 + (i % 5) * 10  # mild variation, purely cosmetic
        image = Image.new("RGB", (200, 200), color=(shade, shade, shade))
        if i < args.defects:
            ImageDraw.Draw(image).rectangle([60, 60, 140, 140], fill=MARKER_RGB)
            name = f"defect_{i:03d}.jpg"
        else:
            name = f"ok_{i:03d}.jpg"
        image.save(raw_dir / name)

    print(f"Generated {args.count} demo image(s) ({args.defects} with a defect marker) "
          f"under '{raw_dir}'.")


if __name__ == "__main__":
    main()
