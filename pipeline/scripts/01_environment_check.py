#!/usr/bin/env python3
"""Stage 1/7: detect the runtime environment, write `outputs/env_report.json`.

Explicitly determines whether this is running on a Jetson Orin Nano
(`/etc/nv_tegra_release`, JetPack's own marker file -- see
`utils.environment.is_jetson`) and logs the L4T/JetPack version, currently
available memory, and current `nvpmodel` power mode. This is the one stage
that *reports* on the device; it deliberately does not itself set
`device.target` in the pipeline config -- that stays an explicit
`--device`/config choice (wired up day 3's `run_all.sh`/Makefile) so a dev
machine can still dry-run the jetson profile without this script pretending
it's actually a Jetson.

Input: none (reads the live machine, not `data/`).
Output: `<outputs_dir>/env_report.json`.

Usage:
    python3 scripts/01_environment_check.py [--config config/pipeline_config.yaml]
        [--marker-path /etc/nv_tegra_release]

`--marker-path` overrides which file is checked for Jetson detection --
exists so this script is testable against a fixture file standing in for
`/etc/nv_tegra_release` (see `tests/test_01_environment_check.py`), not
something you'd normally pass on a real device.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import load_config  # noqa: E402
from utils.environment import DEFAULT_NV_TEGRA_RELEASE_PATH, build_environment_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `01_environment_check.py`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/pipeline_config.yaml")
    parser.add_argument(
        "--marker-path",
        default=str(DEFAULT_NV_TEGRA_RELEASE_PATH),
        help="Path checked for Jetson detection (default: /etc/nv_tegra_release). "
        "Override only for testing.",
    )
    return parser.parse_args()


def main() -> None:
    """Build the environment report and write it to `outputs/env_report.json`."""
    args = parse_args()
    config = load_config(args.config)

    report = build_environment_report(marker_path=Path(args.marker_path))
    report_dict = asdict(report)

    outputs_dir = config.paths.outputs_dir
    outputs_dir.mkdir(parents=True, exist_ok=True)
    output_path = outputs_dir / "env_report.json"
    output_path.write_text(json.dumps(report_dict, indent=2))

    print(json.dumps(report_dict, indent=2))
    print(f"Environment report written to '{output_path}'.")
    if not report.is_jetson:
        print(
            f"Not running on a Jetson (no '{args.marker_path}') -- "
            "l4t_version/jetpack_generation/power_mode are all null, as expected off-device.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
