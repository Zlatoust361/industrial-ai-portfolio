#!/usr/bin/env python3
"""Week 25's whole non-interactive deliverable, one command.

Builds a deterministic synthetic dataset from week 05's real defect-mask
generator (`classical_cv.defect_images`), runs the real OpenCV classical-CV
detector (`classical_cv.detectors`) over it, measures accuracy/latency
(`classical_cv.benchmark`), attempts the real CLIP zero-shot baseline
(`classical_cv.clip_baseline`, honestly reported as `blocked` if this
sandbox's network boundary is hit), and writes one combined comparison
report (`classical_cv.report`).

Usage:
    PYTHONPATH=. python3 classical_cv/scripts/run_classical_cv_demo.py \\
        [--num-positive 30] [--num-negative 30] [--seed 42] [--skip-clip]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_CLASSICAL_CV_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CLASSICAL_CV_ROOT.parent))

from classical_cv.benchmark import evaluate_classical_cv  # noqa: E402
from classical_cv.clip_baseline import run_clip_baseline_or_blocked  # noqa: E402
from classical_cv.defect_images import build_demo_dataset  # noqa: E402
from classical_cv.report import CLIP_STATUS_BLOCKED, ClipBaselineSummary, build_report, to_dict  # noqa: E402
from src.synthetic.config import load_defects_config  # noqa: E402

DEFAULT_OUTPUT_PATH = "outputs/classical_cv_demo/report.json"
DEFAULT_DEFECTS_CONFIG_PATH = "config/defects.yaml"
DEFAULT_NUM_POSITIVE = 30
DEFAULT_NUM_NEGATIVE = 30
DEFAULT_SEED = 42


def run_classical_cv_demo(
    num_positive: int,
    num_negative: int,
    seed: int,
    defects_config_path: str,
    skip_clip: bool,
) -> dict[str, object]:
    """Run the full week 25 comparison and return a JSON-serializable report.

    Args:
        num_positive: Images with 1-2 real defects each.
        num_negative: Clean images (false-positive measurement).
        seed: Dataset RNG seed -- reproducible across runs.
        defects_config_path: Path to `config/defects.yaml` (week 05's real shapes).
        skip_clip: Skip the CLIP baseline attempt entirely (faster
            iteration -- it's expected to be `blocked` in this sandbox
            anyway, see `classical_cv/clip_baseline.py`'s module docstring).

    Returns:
        `to_dict(build_report(...))`'s output.
    """
    defects_config = load_defects_config(defects_config_path)
    images = build_demo_dataset(defects_config, num_positive, num_negative, seed)

    classical_report = evaluate_classical_cv(images)
    if skip_clip:
        clip_summary = ClipBaselineSummary(
            status=CLIP_STATUS_BLOCKED, blocked_reason="--skip-clip was passed"
        )
    else:
        clip_summary = run_clip_baseline_or_blocked(images)

    return to_dict(build_report(classical_report, clip_summary))


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `run_classical_cv_demo.py`."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--num-positive", type=int, default=DEFAULT_NUM_POSITIVE)
    parser.add_argument("--num-negative", type=int, default=DEFAULT_NUM_NEGATIVE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--defects-config", default=DEFAULT_DEFECTS_CONFIG_PATH)
    parser.add_argument(
        "--skip-clip",
        action="store_true",
        help="Skip the real CLIP baseline attempt (it is expected to hit this "
        "sandbox's network boundary and be reported as blocked anyway).",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    """Run the demo, write the report, print the headline numbers."""
    args = parse_args()
    report = run_classical_cv_demo(
        num_positive=args.num_positive,
        num_negative=args.num_negative,
        seed=args.seed,
        defects_config_path=args.defects_config,
        skip_clip=args.skip_clip,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    classical = report["classical_cv"]
    clip = report["clip_baseline"]
    print(f"Classical CV: AP={classical['average_precision']:.3f} "
          f"mean_latency={classical['mean_latency_ms']:.2f}ms")
    print(f"CLIP baseline: status={clip['status']}")
    print(f"Recommendation: {report['recommendation']}")
    print(f"Full report written to {output_path}")


if __name__ == "__main__":
    main()
