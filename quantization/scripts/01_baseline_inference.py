#!/usr/bin/env python3
"""Step 1/8: FP16 baseline latency/throughput, the recipe's starting point.

"baseline FP16 -> calibration -> quantization -> TensorRT engine -> FP16 vs
INT4 measurement -> final report" (this week's own framing, see
`../README.md`) starts here: run the plain HuggingFace CLIP checkpoint
(reusing week 01's `src.models.clip.CLIPLoader`/`CLIPInference` directly,
not reimplemented) at `model.precision` (FP16 by default) and record
latency/throughput across `baseline.batch_sizes` -- the number every later
step (`06_trt_inference.py`, day 3) compares its INT4 numbers against.

Not gated to the Jetson: a FP16 PyTorch baseline is portable, and running
it on a dev machine first (to sanity-check the recipe before burning time
on-device) is the whole point of keeping steps 1-3 device-agnostic (see
`../README.md`'s "Why steps 1-3 aren't gated" section).

Input: none (a synthetic all-zero image, or `--image` for a real one).
Output: `<outputs_dir>/baseline/baseline_results.json`.

Usage:
    python3 scripts/01_baseline_inference.py [--config config/config.yaml] [--image PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

_QUANTIZATION_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_QUANTIZATION_ROOT))
# CLIPLoader/CLIPInference/benchmark_image_encoder are week 01/07's already-tested
# code, one level up from quantization/ -- see this module's docstring.
sys.path.insert(0, str(_QUANTIZATION_ROOT.parent))

from utils.config import QuantizationConfig, load_config  # noqa: E402


def build_inference(config: QuantizationConfig):
    """Build a `CLIPInference` from this week's own `model:`/`memory:` config sections.

    Args:
        config: A loaded `QuantizationConfig`.

    Returns:
        A `src.models.clip.CLIPInference`, model loaded lazily on first call.
    """
    from src.config import MemorySection, ModelSection
    from src.models.clip import CLIPInference, CLIPLoader

    model_section = ModelSection(
        name=config.model.name,
        type=config.model.type,
        precision=config.model.precision,
        device=config.model.device,
        seed=config.model.seed,
        cache_dir=config.model.cache_dir,
    )
    memory_section = MemorySection(
        memory_budget_mb=config.memory.memory_budget_mb,
        device_total_mb=config.memory.device_total_mb,
    )
    return CLIPInference(CLIPLoader(model_section, memory_section))


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `01_baseline_inference.py`."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--image", default=None, help="Sample image path; a synthetic one is used if omitted."
    )
    return parser.parse_args()


def main() -> None:
    """Benchmark the FP16 baseline CLIP image encoder and save a JSON report."""
    args = parse_args()
    config = load_config(args.config)

    from PIL import Image

    from src.evaluation.benchmark import benchmark_image_encoder

    inference = build_inference(config)
    sample_image = (
        Image.open(args.image).convert("RGB")
        if args.image
        else Image.new("RGB", (config.baseline.image_size, config.baseline.image_size))
    )

    stats = benchmark_image_encoder(
        inference,
        sample_image,
        batch_sizes=config.baseline.batch_sizes,
        warmup_iters=config.baseline.warmup_iters,
        num_iters=config.baseline.num_iters,
    )

    report = {
        "metadata": {
            "model_name": config.model.name,
            "precision": config.model.precision,
            "device": config.model.device,
            "image_size": config.baseline.image_size,
        },
        "results": [asdict(entry) for entry in stats],
    }

    output_dir = config.paths.outputs_dir / "baseline"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "baseline_results.json"
    output_path.write_text(json.dumps(report, indent=2))
    print(f"Baseline ({config.model.precision}) report written to '{output_path}'.")


if __name__ == "__main__":
    main()
