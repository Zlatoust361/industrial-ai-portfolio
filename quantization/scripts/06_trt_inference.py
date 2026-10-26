#!/usr/bin/env python3
"""Step 6/8: FP16-engine vs INT4-engine latency, ON the Jetson, at a fixed nvpmodel power mode.

MUST run directly on the Jetson Orin Nano (see `utils.environment.require_jetson`):
loads and runs an actual `.engine` file, which only deserializes on the
exact GPU architecture/TensorRT version that built it -- same constraint
every TensorRT-touching script this week already documents.

Run this **twice per power mode** -- once per engine -- and twice more for
the second power mode, so both the FP16-vs-INT4 comparison and the
15W-vs-MAXN-Super comparison come from real, separately-measured runs
under an explicitly fixed regime, not an assumption:

    # 15W: the standard operating power mode for a 24/7 deployment
    python3 scripts/06_trt_inference.py --engine outputs/engine/image_encoder_fp16.engine \\
        --label fp16 --nvpmodel-mode 1
    python3 scripts/06_trt_inference.py --engine outputs/engine/image_encoder.engine \\
        --label int4 --nvpmodel-mode 1

    # MAXN Super: the upper bound, no thermal throttling assumed away
    python3 scripts/06_trt_inference.py --engine outputs/engine/image_encoder_fp16.engine \\
        --label fp16 --nvpmodel-mode 0
    python3 scripts/06_trt_inference.py --engine outputs/engine/image_encoder.engine \\
        --label int4 --nvpmodel-mode 0

`--nvpmodel-mode` is optional: omit it to benchmark under whatever mode
the device is already in (recording it either way via `nvpmodel -q`) --
useful if an operator already set the mode once and wants to run all four
combinations without re-switching (and re-settling) each time.

Input: `--engine` (a `.engine` file from `05_build_engine.py`),
       `<evaluation.test_set_dir>` (week 06's held-out test split).
Output: `<outputs_dir>/trt_inference/<label>_<mode_name>.json`.

Usage (on the Jetson):
    python3 scripts/06_trt_inference.py --engine PATH --label fp16|int4 [--nvpmodel-mode 0|1|2]
        [--config config/config.yaml] [--profile orin_nano_8gb]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

_QUANTIZATION_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_QUANTIZATION_ROOT))
sys.path.insert(0, str(_QUANTIZATION_ROOT.parent))

from utils.config import QuantizationConfig, load_config  # noqa: E402
from utils.environment import DEFAULT_NV_TEGRA_RELEASE_PATH, require_jetson  # noqa: E402
from utils.power_mode import TegrastatsMonitor, query_power_mode, set_nvpmodel  # noqa: E402
from utils.test_set import discover_test_images  # noqa: E402
from utils.trt_inference import TrtImageEncoder, benchmark_trt_encoder  # noqa: E402

TEGRASTATS_INTERVAL_MS = 500
SETTLE_SECONDS = 3.0


def run_inference_benchmark(
    config: QuantizationConfig,
    engine_path: Path,
    profile_name: str | None,
    test_set_dir: Path | None,
) -> dict[str, object]:
    """Benchmark one engine against the fixed test set, sampling `tegrastats` throughout.

    Args:
        config: A loaded `QuantizationConfig`.
        engine_path: The `.engine` file to benchmark.
        profile_name: Builder profile name -- must match the profile the
            engine was actually built with (its fixed batch size).
        test_set_dir: Overrides `evaluation.test_set_dir`.

    Returns:
        `{"latency": {...LatencyStats}, "temperatures": {"mean_temps_c", "max_temps_c"}}`.
    """
    from src.evaluation.jetson import parse_tegrastats_temps

    profile = config.get_profile(profile_name)
    images = discover_test_images(
        test_set_dir or config.evaluation.test_set_dir, config.evaluation.num_samples
    )

    encoder = TrtImageEncoder(engine_path, profile.batch_size, config.baseline.image_size)
    with TegrastatsMonitor(TEGRASTATS_INTERVAL_MS) as monitor:
        stats = benchmark_trt_encoder(
            encoder, images, config.evaluation.warmup_iters, config.evaluation.num_iters
        )
    temp_readings = parse_tegrastats_temps(monitor.lines)

    return {
        "latency": asdict(stats),
        "temperatures": {
            "mean_temps_c": {
                sensor: sum(values) / len(values) for sensor, values in temp_readings.items()
            },
            "max_temps_c": {sensor: max(values) for sensor, values in temp_readings.items()},
        },
        "num_test_images": len(images),
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `06_trt_inference.py`."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--engine", required=True, help="Path to a .engine file to benchmark.")
    parser.add_argument(
        "--label", required=True, help="Tag for this run, e.g. 'fp16' or 'int4' (output filename)."
    )
    parser.add_argument(
        "--profile", default=None, help="Builder profile name; must match the engine's build."
    )
    parser.add_argument(
        "--nvpmodel-mode",
        type=int,
        default=None,
        choices=[0, 1, 2],
        help="Switch to this nvpmodel mode (0=MAXN_SUPER, 1=15W, 2=7W) before benchmarking. "
        "Omit to use whatever mode the device is already in.",
    )
    parser.add_argument("--settle-seconds", type=float, default=SETTLE_SECONDS)
    parser.add_argument("--test-set-dir", default=None, help="Overrides evaluation.test_set_dir.")
    parser.add_argument("--output", default=None, help="Overrides the default output path.")
    parser.add_argument(
        "--marker-path",
        default=str(DEFAULT_NV_TEGRA_RELEASE_PATH),
        help="Path checked for Jetson detection. Override only for testing.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip the Jetson-device check (the resulting numbers won't be meaningful).",
    )
    return parser.parse_args()


def main() -> None:
    """Gate on Jetson, optionally switch power mode, benchmark, and write the report."""
    args = parse_args()
    require_jetson(Path(args.marker_path), "06_trt_inference.py", args.force)

    config = load_config(args.config)
    if args.nvpmodel_mode is not None:
        set_nvpmodel(args.nvpmodel_mode, args.settle_seconds)

    power_mode = query_power_mode()
    mode_name = (power_mode or {}).get("mode_name") or "unknown"

    result = run_inference_benchmark(
        config,
        Path(args.engine),
        args.profile,
        Path(args.test_set_dir) if args.test_set_dir else None,
    )
    report = {
        "metadata": {
            "label": args.label,
            "engine_path": args.engine,
            "profile": args.profile or config.active_profile,
            "power_mode": power_mode,
        },
        **result,
    }

    output_path = (
        Path(args.output)
        if args.output
        else config.paths.outputs_dir / "trt_inference" / f"{args.label}_{mode_name}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    print(
        f"TensorRT inference report ('{args.label}', power mode '{mode_name}') "
        f"written to '{output_path}'."
    )


if __name__ == "__main__":
    main()
