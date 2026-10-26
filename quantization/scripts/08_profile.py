#!/usr/bin/env python3
"""Step 8/8: aggregate every day 3 JSON report into one `summary.md`.

NOT Jetson-gated, unlike steps 4-7: this only reads already-written JSON
(`06_trt_inference.py`'s per-engine/power-mode reports,
`07_evaluate_quality.py`'s quality report, optionally
`01_baseline_inference.py`'s FP16 PyTorch baseline) and renders Markdown --
pure aggregation, portable anywhere, the same way week 07's
`compare_dr_branches.py` could compute real statistics from
already-computed `detections.json` without needing `ultralytics` itself.

Produces:
- **Speedup**: FP16 → INT4 mean/p95 latency ratio, per power mode.
- **Trade-off table**: latency + throughput + quality (cosine similarity
  to FP16) side by side, one row per engine/power-mode combination.
- **Temperature/throttling**: peak `tegrastats` reading per sensor across
  every run in the whole test cycle (not just one run), flagged against
  `evaluation.throttle_temp_c`, plus a per-run breakdown.

Input: `<outputs_dir>/trt_inference/*.json` (required, from `06`),
       `<outputs_dir>/quality/quality_report.json` (optional, from `07`),
       `<outputs_dir>/baseline/baseline_results.json` (optional, from `01`).
Output: `<outputs_dir>/summary.md`.

Usage:
    python3 scripts/08_profile.py [--config config/config.yaml]
        [--inference-dir PATH] [--quality-report PATH] [--baseline-report PATH]
        [--output PATH]
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

_QUANTIZATION_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_QUANTIZATION_ROOT))

from utils.config import QuantizationConfig, load_config  # noqa: E402
from utils.profile_report import (  # noqa: E402
    aggregate_peak_temperatures,
    compute_speedup_rows,
    discover_trt_inference_reports,
    group_by_power_mode,
    load_optional_json,
    render_summary_markdown,
)


def build_summary(
    config: QuantizationConfig,
    inference_dir: Path,
    quality_report_path: Path,
    baseline_report_path: Path,
    generated_at: str,
) -> str:
    """Gather every input and render the full `summary.md` text.

    Args:
        config: A loaded `QuantizationConfig`.
        inference_dir: `06_trt_inference.py`'s output directory.
        quality_report_path: `07_evaluate_quality.py`'s output path (optional).
        baseline_report_path: `01_baseline_inference.py`'s output path (optional).
        generated_at: ISO-8601 timestamp to stamp the report with.

    Returns:
        The rendered Markdown text.

    Raises:
        FileNotFoundError: If `inference_dir` has no reports (see
            `discover_trt_inference_reports`) -- the one required input.
    """
    reports = discover_trt_inference_reports(inference_dir)
    grouped = group_by_power_mode(reports)
    speedup_rows = compute_speedup_rows(grouped)
    peak_temps_c = aggregate_peak_temperatures(reports)

    return render_summary_markdown(
        reports=reports,
        grouped=grouped,
        speedup_rows=speedup_rows,
        quality_report=load_optional_json(quality_report_path),
        baseline_report=load_optional_json(baseline_report_path),
        peak_temps_c=peak_temps_c,
        throttle_temp_c=config.evaluation.throttle_temp_c,
        generated_at=generated_at,
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `08_profile.py`."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--inference-dir", default=None, help="Overrides <outputs_dir>/trt_inference/."
    )
    parser.add_argument(
        "--quality-report",
        default=None,
        help="Overrides <outputs_dir>/quality/quality_report.json.",
    )
    parser.add_argument(
        "--baseline-report",
        default=None,
        help="Overrides <outputs_dir>/baseline/baseline_results.json.",
    )
    parser.add_argument("--output", default=None, help="Overrides <outputs_dir>/summary.md.")
    return parser.parse_args()


def main() -> None:
    """Aggregate every available JSON report and write `summary.md`."""
    args = parse_args()
    config = load_config(args.config)

    inference_dir = (
        Path(args.inference_dir)
        if args.inference_dir
        else config.paths.outputs_dir / "trt_inference"
    )
    quality_report_path = (
        Path(args.quality_report)
        if args.quality_report
        else config.paths.outputs_dir / "quality" / "quality_report.json"
    )
    baseline_report_path = (
        Path(args.baseline_report)
        if args.baseline_report
        else config.paths.outputs_dir / "baseline" / "baseline_results.json"
    )
    output_path = Path(args.output) if args.output else config.paths.outputs_dir / "summary.md"

    summary = build_summary(
        config,
        inference_dir,
        quality_report_path,
        baseline_report_path,
        datetime.now(UTC).isoformat(),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(summary)
    print(f"Summary written to '{output_path}'.")


if __name__ == "__main__":
    main()
