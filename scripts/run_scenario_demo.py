"""CLI: run color_sorting.py / defect_detection.py for N cycles against a real ScenarioContext.

Uses `src.scenarios.build_default_context` (real `VisionBridge`/
`ROSInterface`/`OPCUABridge`/`Manipulator`/`Database`, `simulate=True`
defaults) and the requested scenario's env-aware config
(`config/scenarios.yaml`) + synthetic (dev) or real-camera (jetson)
image source. Every cycle's `ScenarioResult` is printed and saved.

This sandbox has no `torch`, so the vision service's `/v1/embed/*` calls
will fail with a real `VisionBridgeError` here (nothing is listening on
`vision_bridge.base_url` either) -- every result below is expected to
report a clean classification failure, not a crash. That's the honest,
verified-to-the-real-boundary state this script proves, the same
pattern `assistant/scripts/run_action_demo.py` (week 11) established.

Usage:
    PYTHONPATH=. python scripts/run_scenario_demo.py color_sorting \\
        [--cycles 3] [--env dev] [--output-dir outputs/scenarios_demo]
    PYTHONPATH=. python scripts/run_scenario_demo.py defect_detection \\
        [--cycles 3] [--env dev]
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

from src.config.environment import Environment
from src.logging import configure_logging
from src.scenarios import ColorSortingScenario, DefectDetectionScenario, build_default_context
from src.scenarios.config import load_color_sorting_config, load_defect_detection_config
from src.scenarios.image_sources import (
    build_color_sorting_image_source,
    build_defect_detection_image_source,
)

SCENARIO_NAMES = ("color_sorting", "defect_detection")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `run_scenario_demo.py`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", choices=SCENARIO_NAMES)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument(
        "--env", choices=[e.value for e in Environment], default=Environment.DEV.value
    )
    parser.add_argument("--output-dir", default="outputs/scenarios_demo")
    return parser.parse_args()


def _build_scenario(name: str, environment: Environment):
    if name == "color_sorting":
        config = load_color_sorting_config(environment=environment)
        image_source = build_color_sorting_image_source(config.camera, config.colors)
        return ColorSortingScenario(config=config, image_source=image_source)

    config = load_defect_detection_config(environment=environment)
    image_source = build_defect_detection_image_source(config.camera)
    return DefectDetectionScenario(config=config, image_source=image_source)


def main() -> None:
    """Run `args.cycles` cycles of `args.scenario`, print + save each `ScenarioResult`."""
    args = parse_args()
    configure_logging(logging.INFO)
    environment = Environment(args.env)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario = _build_scenario(args.scenario, environment)
    results = []
    with build_default_context(environment, db_path=str(output_dir / "detections.db")) as ctx:
        for cycle in range(args.cycles):
            result = scenario.run(ctx)
            results.append(asdict(result))
            print(f"[{cycle + 1}/{args.cycles}] success={result.success} message={result.message}")

    output_path = output_dir / f"{args.scenario}_results.json"
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    print(f"Results written to '{output_path}'.")


if __name__ == "__main__":
    main()
