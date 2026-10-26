#!/usr/bin/env python3
"""Step 4/8: run TensorRT INT8 entropy calibration over the calibration set, ON the Jetson.

MUST run directly on the Jetson Orin Nano (see `utils.environment.require_jetson`):
`tensorrt`'s calibrator interface allocates device memory and runs the
network's kernels to collect per-layer activation ranges -- both the
calibration numbers and the resulting cache are tied to the exact GPU
architecture/TensorRT build that produced them, not portable off-device
any more than a `.engine` file is (same constraint week 03's
`convert_owlvit_tensorrt.py` and week 07's `export_and_compare_jetson.py`
document for the engine itself). Refuses to run elsewhere by default;
`--force` overrides, for exercising this script's argument handling on a
dev machine -- it will still fail importing `tensorrt` there (not
pip-installable off-JetPack, see `../README.md`), which is the honest
expected outcome of forcing it.

This step runs a real (its serialized engine is discarded) TensorRT
builder pass purely to drive calibration: TensorRT's `IInt8EntropyCalibrator2`
is bit-width-agnostic despite the "Int8" name -- the same calibration cache
this writes is what `05_build_engine.py` reuses regardless of whether
`quantization.bits` is 4 or 8 (the bit-width itself is a separate builder
flag, applied in step 5's real, kept build). This keeps calibration on
TensorRT's oldest, most stable API instead of a newer INT4-specific path
this repo can't verify against real hardware either way -- see
`../utils/tensorrt_builder.py`'s module docstring.

Input: `<paths.calibration_data_dir>/manifest.json` (from `03_prepare_calibration_data.py`),
       `<paths.onnx_path>` (from `02_export_onnx.py`).
Output: `<paths.calibration_cache_path>`.

Usage (on the Jetson):
    python3 scripts/04_quantize_model.py [--config config/config.yaml] [--profile orin_nano_8gb]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_QUANTIZATION_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_QUANTIZATION_ROOT))

from utils.config import QuantizationConfig, load_config  # noqa: E402
from utils.environment import DEFAULT_NV_TEGRA_RELEASE_PATH, require_jetson  # noqa: E402
from utils.tensorrt_builder import (  # noqa: E402
    build_calibrator,
    configure_builder_config,
    load_calibration_manifest,
    parse_onnx_network,
)

CALIBRATION_BITS = 8
"""Always calibrate at INT8-entropy semantics -- see this module's docstring."""


def run_calibration(config: QuantizationConfig, profile_name: str | None) -> Path:
    """Run a (discarded) TensorRT build pass purely to produce the calibration cache.

    Args:
        config: A loaded `QuantizationConfig`.
        profile_name: Builder profile name (see `QuantizationConfig.get_profile`).

    Returns:
        Path to the written calibration cache.

    Raises:
        FileNotFoundError: If the calibration manifest or ONNX file is missing.
        RuntimeError: If the ONNX parse or the build pass itself fails.
    """
    profile = config.get_profile(profile_name)
    image_paths = load_calibration_manifest(config.paths.calibration_data_dir)
    builder, network, _parser = parse_onnx_network(config.paths.onnx_path)

    calibrator = build_calibrator(
        image_paths,
        batch_size=profile.batch_size,
        image_size=config.calibration.image_size,
        cache_path=config.paths.calibration_cache_path,
    )
    builder_config = configure_builder_config(
        builder,
        network,
        bits=CALIBRATION_BITS,
        workspace_size_mb=profile.workspace_size_mb,
        fp16_fallback=profile.fp16_fallback,
        batch_size=profile.batch_size,
        calibrator=calibrator,
    )

    # The serialized engine itself is discarded -- this step's only real
    # output is the calibration cache, written as a side effect of the
    # build pass calling the calibrator's get_batch()/write_calibration_cache().
    serialized_engine = builder.build_serialized_network(network, builder_config)
    if serialized_engine is None:
        raise RuntimeError(
            "TensorRT calibration build pass failed (build_serialized_network returned None) -- "
            "check the builder log above for the underlying error."
        )

    if not config.paths.calibration_cache_path.is_file():
        raise RuntimeError(
            f"Calibration build pass succeeded but no cache was written to "
            f"{config.paths.calibration_cache_path} -- this shouldn't happen; check "
            "ImageEntropyCalibrator.write_calibration_cache in utils/tensorrt_builder.py."
        )
    return config.paths.calibration_cache_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `04_quantize_model.py`."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--profile", default=None, help="Builder profile name; defaults to config's active_profile."
    )
    parser.add_argument(
        "--marker-path",
        default=str(DEFAULT_NV_TEGRA_RELEASE_PATH),
        help="Path checked for Jetson detection. Override only for testing.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip the Jetson-device check (the resulting cache won't be meaningful).",
    )
    return parser.parse_args()


def main() -> None:
    """Gate on Jetson, then run TensorRT calibration and write the cache."""
    args = parse_args()
    require_jetson(Path(args.marker_path), "04_quantize_model.py", args.force)

    config = load_config(args.config)
    profile = config.get_profile(args.profile)
    print(f"Using profile '{args.profile or config.active_profile}': {profile.description}")

    cache_path = run_calibration(config, args.profile)
    print(f"Calibration cache written to '{cache_path}'.")


if __name__ == "__main__":
    main()
