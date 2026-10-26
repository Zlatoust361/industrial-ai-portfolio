"""Run the configured latency/throughput benchmark and write a JSON report.

Usage:
    PYTHONPATH=. python scripts/03_benchmark.py [--config config/model_config.yaml] \\
        [--image PATH] [--target dev|jetson]

`--target jetson` forces fp16 precision (the on-device deployment path) and,
if this process is actually running on a Jetson (i.e. `nvpmodel` is
present), records the device's current power mode in the report's
`metadata.power_mode`. On a dev machine `nvpmodel` won't exist, so
`power_mode` is simply left `null` -- the flag doesn't require real
hardware to use.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict

from PIL import Image

from src.config import apply_target_overrides, load_config
from src.evaluation.benchmark import benchmark_image_encoder
from src.evaluation.jetson import parse_nvpmodel_query
from src.models.clip import CLIPInference, CLIPLoader


def query_current_power_mode() -> dict[str, str | int | None] | None:
    """Query `nvpmodel -q` for the device's current power mode, if available.

    Returns:
        The parsed `{"mode_name", "mode_id"}` mapping, or `None` if
        `nvpmodel` isn't present/runnable on this machine (i.e. this isn't
        running on a Jetson device).
    """
    try:
        result = subprocess.run(
            ["nvpmodel", "-q"], capture_output=True, text=True, timeout=5, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_nvpmodel_query(result.stdout)


def main() -> None:
    """Benchmark the configured CLIP image encoder and save a JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/model_config.yaml")
    parser.add_argument(
        "--image", default=None, help="Sample image path; a synthetic one is used if omitted."
    )
    parser.add_argument(
        "--target",
        choices=["dev", "jetson"],
        default="dev",
        help="'jetson' forces fp16 and logs the current nvpmodel power mode in metadata.",
    )
    args = parser.parse_args()

    config = apply_target_overrides(load_config(args.config), args.target)
    loader = CLIPLoader(config.model, config.memory)
    inference = CLIPInference(loader)

    sample_image = (
        Image.open(args.image).convert("RGB")
        if args.image
        else Image.new("RGB", (config.inference.image_size, config.inference.image_size))
    )

    stats = benchmark_image_encoder(
        inference,
        sample_image,
        batch_sizes=config.benchmark.batch_sizes,
        warmup_iters=config.benchmark.warmup_iters,
        num_iters=config.benchmark.num_iters,
    )

    power_mode = query_current_power_mode() if args.target == "jetson" else None
    report = {
        "metadata": {
            "target": args.target,
            "model_name": config.model.name,
            "precision": config.model.precision,
            "power_mode": power_mode,
        },
        "results": [asdict(entry) for entry in stats],
    }

    output_dir = config.benchmark.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "latency_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Benchmark report written to '{report_path}'.")


if __name__ == "__main__":
    main()
