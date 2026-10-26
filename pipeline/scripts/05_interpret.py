#!/usr/bin/env python3
"""Stage 5/7: turn raw detection scores into ok/defect verdicts.

Writes `outputs/interpretations.json`.

Thin CLI wrapper around `utils.interpret.interpret_detections` -- all the
actual decision logic lives there (pure, unit-tested), so this script is
just I/O glue: read `outputs/detections.json`, apply
`interpret.defect_score_threshold`, write the result.

Input: `outputs/detections.json` (from `04_detect.py`).
Output: `outputs/interpretations.json`.

Usage:
    python3 scripts/05_interpret.py [--config config/pipeline_config.yaml]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import load_config  # noqa: E402
from utils.interpret import interpret_detections  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `05_interpret.py`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/pipeline_config.yaml")
    return parser.parse_args()


def main() -> None:
    """Classify every detection as ok/defect and write `outputs/interpretations.json`."""
    args = parse_args()
    config = load_config(args.config)

    detections_path = config.paths.outputs_dir / "detections.json"
    if not detections_path.is_file():
        print(
            f"Error: detections not found at '{detections_path}' -- run 04_detect.py first.",
            file=sys.stderr,
        )
        sys.exit(1)
    detections = json.loads(detections_path.read_text())["detections"]

    verdicts, summary = interpret_detections(detections, config.interpret.defect_score_threshold)

    interpretations_path = config.paths.outputs_dir / "interpretations.json"
    interpretations_path.write_text(
        json.dumps(
            {
                "defect_score_threshold": config.interpret.defect_score_threshold,
                "summary": summary,
                "verdicts": [verdict.to_dict() for verdict in verdicts],
            },
            indent=2,
        )
    )

    print(f"Summary: {summary}")
    print(f"Interpretations written to '{interpretations_path}'.")


if __name__ == "__main__":
    main()
