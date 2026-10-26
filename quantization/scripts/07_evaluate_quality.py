#!/usr/bin/env python3
"""Step 7/8: FP16-engine vs INT4-engine embedding quality, ON the Jetson.

MUST run directly on the Jetson Orin Nano (see `utils.environment.require_jetson`):
same reasoning as `06_trt_inference.py` -- both `.engine` files only
deserialize on the exact hardware/TensorRT version that built them.

Quality metric: per-image cosine similarity between the FP16 engine's and
the quantized engine's image embeddings, on the exact same fixed test set
`06_trt_inference.py` uses (`evaluation.test_set_dir`, same deterministic
sort-and-cap via `utils.test_set.discover_test_images` -- not a
downstream classification accuracy, since this repo's CLIP model is a
zero-shot *embedding* encoder, not a task-specific classifier with its own
labels to check against). A real gate, like week 04/06's
`06_quality_gate.py`/`08_validate_dataset.py`: exits `1` if the mean
similarity drops below `evaluation.min_cosine_similarity`.

Like `06_trt_inference.py`, accepts `--nvpmodel-mode` for consistency (a
fixed power regime is this day's ask for the whole FP16-vs-INT4
comparison) even though embedding quality itself shouldn't depend on
clock speed -- recorded in the report either way, in case it ever doesn't.

Input: `<paths.fp16_engine_path>`/`<paths.engine_path>` (from
       `05_build_engine.py`, both builds), `<evaluation.test_set_dir>`.
Output: `<outputs_dir>/quality/quality_report.json`.

Usage (on the Jetson, after both engine builds exist):
    python3 scripts/07_evaluate_quality.py [--config config/config.yaml] [--profile orin_nano_8gb]
        [--nvpmodel-mode 0|1|2]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_QUANTIZATION_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_QUANTIZATION_ROOT))
# utils.power_mode reuses src.evaluation.jetson (repo root, one level up) --
# see this module's docstring / utils/power_mode.py's own.
sys.path.insert(0, str(_QUANTIZATION_ROOT.parent))

from utils.config import QuantizationConfig, load_config  # noqa: E402
from utils.environment import DEFAULT_NV_TEGRA_RELEASE_PATH, require_jetson  # noqa: E402
from utils.power_mode import query_power_mode, set_nvpmodel  # noqa: E402
from utils.test_set import discover_test_images  # noqa: E402
from utils.trt_inference import TrtImageEncoder, cosine_similarity  # noqa: E402

SETTLE_SECONDS = 3.0


def evaluate_quality(
    config: QuantizationConfig, profile_name: str | None, test_set_dir: Path | None
) -> dict[str, object]:
    """Encode the fixed test set with both engines and compare embeddings.

    Args:
        config: A loaded `QuantizationConfig`.
        profile_name: Builder profile name -- must match the profile both
            engines were actually built with (their fixed batch size).
        test_set_dir: Overrides `evaluation.test_set_dir`.

    Returns:
        `{"num_images", "mean_cosine_similarity", "min_cosine_similarity",
        "max_cosine_similarity", "num_below_threshold", "threshold",
        "passed", "per_image"}`.

    Raises:
        FileNotFoundError: If either engine, or the test-set directory, is missing.
    """
    profile = config.get_profile(profile_name)
    images = discover_test_images(
        test_set_dir or config.evaluation.test_set_dir, config.evaluation.num_samples
    )

    fp16_encoder = TrtImageEncoder(
        config.paths.fp16_engine_path, profile.batch_size, config.baseline.image_size
    )
    quant_encoder = TrtImageEncoder(
        config.paths.engine_path, profile.batch_size, config.baseline.image_size
    )

    similarities: list[float] = []
    per_image: list[dict[str, object]] = []
    batch_size = profile.batch_size
    for start in range(0, len(images) - len(images) % batch_size, batch_size):
        batch = images[start : start + batch_size]
        fp16_embeds = fp16_encoder.encode(batch)
        quant_embeds = quant_encoder.encode(batch)
        batch_similarities = cosine_similarity(fp16_embeds, quant_embeds)
        for image_path, similarity in zip(batch, batch_similarities, strict=True):
            similarities.append(float(similarity))
            per_image.append({"image": str(image_path), "cosine_similarity": float(similarity)})

    threshold = config.evaluation.min_cosine_similarity
    mean_similarity = sum(similarities) / len(similarities) if similarities else 0.0
    return {
        "num_images": len(similarities),
        "mean_cosine_similarity": mean_similarity,
        "min_cosine_similarity": min(similarities) if similarities else 0.0,
        "max_cosine_similarity": max(similarities) if similarities else 0.0,
        "num_below_threshold": sum(1 for value in similarities if value < threshold),
        "threshold": threshold,
        "passed": mean_similarity >= threshold,
        "per_image": per_image,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `07_evaluate_quality.py`."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--profile", default=None, help="Builder profile name; must match both engines' build."
    )
    parser.add_argument(
        "--nvpmodel-mode",
        type=int,
        default=None,
        choices=[0, 1, 2],
        help="Switch to this nvpmodel mode before evaluating. Omit to use the device's "
        "current mode.",
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
    """Gate on Jetson, optionally switch power mode, evaluate quality, and write the report."""
    args = parse_args()
    require_jetson(Path(args.marker_path), "07_evaluate_quality.py", args.force)

    config = load_config(args.config)
    if args.nvpmodel_mode is not None:
        set_nvpmodel(args.nvpmodel_mode, args.settle_seconds)
    power_mode = query_power_mode()

    result = evaluate_quality(
        config, args.profile, Path(args.test_set_dir) if args.test_set_dir else None
    )
    report = {
        "metadata": {
            "profile": args.profile or config.active_profile,
            "power_mode": power_mode,
            "fp16_engine_path": str(config.paths.fp16_engine_path),
            "quantized_engine_path": str(config.paths.engine_path),
            "quantization_bits": config.quantization.bits,
        },
        **result,
    }

    output_path = (
        Path(args.output)
        if args.output
        else config.paths.outputs_dir / "quality" / "quality_report.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))

    status = "PASSED" if result["passed"] else "FAILED"
    print(
        f"Quality gate {status}: mean cosine similarity "
        f"{result['mean_cosine_similarity']:.4f} (threshold {result['threshold']}), "
        f"{result['num_below_threshold']}/{result['num_images']} images below threshold. "
        f"Report written to '{output_path}'."
    )
    if not result["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
