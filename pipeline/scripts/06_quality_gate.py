#!/usr/bin/env python3
"""Stage 6/7: GREEN/YELLOW/RED quality gate, write `outputs/metrics/quality_gate.json`.

Thin CLI wrapper around `utils.quality_gate.evaluate_quality_gate` -- the
threshold logic lives there (pure, unit-tested). Writes to
`outputs/metrics/` specifically (not `outputs/` directly) since that's the
directory day 3's GitHub Actions gate is scoped to watch.

Exit code doubles as a simple CI signal: `0` for GREEN/YELLOW, `1` for RED
-- YELLOW passes (with a stderr warning) rather than failing a build, since
it means "within tolerance but worth a human look," not "reject."

Input: `outputs/interpretations.json` (from `05_interpret.py`).
Output: `outputs/metrics/quality_gate.json`.

Usage:
    python3 scripts/06_quality_gate.py [--config config/pipeline_config.yaml]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import load_config  # noqa: E402
from utils.quality_gate import evaluate_quality_gate  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `06_quality_gate.py`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/pipeline_config.yaml")
    return parser.parse_args()


def main() -> None:
    """Evaluate the quality gate and write `outputs/metrics/quality_gate.json`."""
    args = parse_args()
    config = load_config(args.config)

    interpretations_path = config.paths.outputs_dir / "interpretations.json"
    if not interpretations_path.is_file():
        print(
            f"Error: interpretations not found at '{interpretations_path}' -- "
            "run 05_interpret.py first.",
            file=sys.stderr,
        )
        sys.exit(1)
    defect_rate = json.loads(interpretations_path.read_text())["summary"]["defect_rate"]

    result = evaluate_quality_gate(
        defect_rate=defect_rate,
        green_max_defect_rate=config.quality_gate.green_max_defect_rate,
        yellow_max_defect_rate=config.quality_gate.yellow_max_defect_rate,
    )

    metrics_dir = config.paths.outputs_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "quality_gate.json"
    metrics_path.write_text(json.dumps(result.to_dict(), indent=2))

    print(f"Quality gate: {result.status} (defect_rate={result.defect_rate:.3f})")
    print(f"Metrics written to '{metrics_path}'.")

    if result.status == "YELLOW":
        print("Warning: defect rate is within tolerance but elevated.", file=sys.stderr)
    elif result.status == "RED":
        print("Quality gate failed.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
