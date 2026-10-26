#!/usr/bin/env python3
"""Stage 2/7: validate raw images, write `data/manifest.json`.

Reads every file matching `ingest.input_glob` under `<data_dir>/raw/`,
confirms each one actually decodes as an image (via PIL) and records its
dimensions. A file that fails to open is skipped with a warning, not fatal
-- one corrupt image shouldn't block an entire batch's inspection run (see
`utils.manifest.build_manifest`'s docstring) -- but zero *valid* images is
fatal, since every downstream stage needs at least one.

Input: `<data_dir>/raw/*` (glob from config).
Output: `<data_dir>/manifest.json`.

Usage:
    python3 scripts/02_ingest.py [--config config/pipeline_config.yaml]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import load_config  # noqa: E402
from utils.manifest import build_manifest, discover_images  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `02_ingest.py`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/pipeline_config.yaml")
    return parser.parse_args()


def main() -> None:
    """Validate raw images and write `data/manifest.json`."""
    args = parse_args()
    config = load_config(args.config)

    raw_dir = config.paths.data_dir / "raw"
    candidate_paths = discover_images(raw_dir, config.ingest.input_glob)
    records, errors = build_manifest(candidate_paths)

    for error in errors:
        print(f"Skipping unreadable image: {error}", file=sys.stderr)

    if not records:
        print(
            f"Error: no valid images found under '{raw_dir}' matching "
            f"'{config.ingest.input_glob}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    config.paths.data_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = config.paths.data_dir / "manifest.json"
    manifest = {
        "raw_dir": str(raw_dir),
        "input_glob": config.ingest.input_glob,
        "images": [record.to_dict() for record in records],
        "skipped": errors,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"Ingested {len(records)} image(s), skipped {len(errors)}.")
    print(f"Manifest written to '{manifest_path}'.")


if __name__ == "__main__":
    main()
