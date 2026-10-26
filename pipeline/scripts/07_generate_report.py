#!/usr/bin/env python3
"""Stage 7/7: render the final Markdown report, write `outputs/report.md`.

Thin CLI wrapper around `utils.report.render_report` -- the rendering logic
lives there (pure, unit-tested). Combines every upstream stage's artifact
(environment, interpretation summary, quality gate, per-image verdicts)
into one human-readable report.

Input: `outputs/env_report.json`, `outputs/interpretations.json`,
    `outputs/metrics/quality_gate.json`.
Output: `outputs/report.md`.

Usage:
    python3 scripts/07_generate_report.py [--config config/pipeline_config.yaml]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import load_config  # noqa: E402
from utils.report import render_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `07_generate_report.py`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/pipeline_config.yaml")
    return parser.parse_args()


def _load_required(path: Path, produced_by: str) -> dict[str, object]:
    """Load a required upstream JSON artifact, or exit with a clear error if it's missing."""
    if not path.is_file():
        print(f"Error: '{path}' not found -- run {produced_by} first.", file=sys.stderr)
        sys.exit(1)
    return json.loads(path.read_text())


def main() -> None:
    """Render `outputs/report.md` from every upstream stage's artifacts."""
    args = parse_args()
    config = load_config(args.config)
    outputs_dir = config.paths.outputs_dir

    env_report = _load_required(outputs_dir / "env_report.json", "01_environment_check.py")
    interpretations = _load_required(outputs_dir / "interpretations.json", "05_interpret.py")
    quality_gate = _load_required(
        outputs_dir / "metrics" / "quality_gate.json", "06_quality_gate.py"
    )

    report = render_report(
        title=config.report.title,
        env_report=env_report,
        interpret_summary=interpretations["summary"],
        quality_gate=quality_gate,
        verdicts=interpretations["verdicts"],
    )

    report_path = outputs_dir / "report.md"
    report_path.write_text(report)

    print(report)
    print(f"Report written to '{report_path}'.")


if __name__ == "__main__":
    main()
