#!/usr/bin/env python3
"""Step 5/8: serialize the final `.engine` file, ON the Jetson.

MUST run directly on the Jetson Orin Nano (see `utils.environment.require_jetson`):
a `.engine` file is not portable across GPU architectures/TensorRT
versions -- the same hard constraint week 03's
`ros2_ws/scripts/convert_owlvit_tensorrt.py` and week 07's
`scripts/export_and_compare_jetson.py` already document. Building anywhere
else produces a file that won't load where it's meant to run. Refuses to
run elsewhere by default; `--force` overrides, for exercising this
script's argument handling on a dev machine -- it will still fail
importing `tensorrt` there, the honest expected outcome of forcing it.

Two build modes:

- Default: applies `quantization.bits` (4 or 8, from config.yaml) as the
  real builder flag (unlike `04_quantize_model.py`, which always
  calibrates at INT8 semantics regardless of the target bit-width -- see
  that script's docstring), reusing the calibration cache `04` wrote so
  this build doesn't re-run calibration image-by-image if the cache is
  already there. Writes `<paths.engine_path>`.
- `--fp16-only`: a plain FP16 build, no INT4/INT8 flag and no calibration
  cache needed at all. This is the reference engine
  `06_trt_inference.py`/`07_evaluate_quality.py` compare the quantized
  build against -- run this once (in addition to a default run) to get
  both engines. Writes `<paths.fp16_engine_path>`.

Input: `<paths.onnx_path>` (from `02_export_onnx.py`),
       `<paths.calibration_cache_path>` (from `04_quantize_model.py`, default mode only).
Output: `<paths.engine_path>` (default) or `<paths.fp16_engine_path>` (`--fp16-only`).

Usage (on the Jetson):
    python3 scripts/05_build_engine.py [--config config/config.yaml] [--profile orin_nano_8gb]
    python3 scripts/05_build_engine.py --fp16-only [--config config/config.yaml] \\
        [--profile orin_nano_8gb]
    # Real Jetson OOM even on orin_nano_8gb's 1024 MB workspace (low free
    # unified memory at build time -- see --workspace-mb's help):
    python3 scripts/05_build_engine.py --fp16-only --workspace-mb 256
"""

from __future__ import annotations

import argparse
import dataclasses
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


def build_engine(
    config: QuantizationConfig,
    profile_name: str | None,
    fp16_only: bool,
    workspace_mb_override: int | None = None,
) -> Path:
    """Parse the ONNX graph and serialize the final `.engine` file.

    Args:
        config: A loaded `QuantizationConfig`.
        profile_name: Builder profile name (see `QuantizationConfig.get_profile`).
        fp16_only: Build a plain FP16 engine (no calibration cache needed)
            instead of the `quantization.bits`-quantized one.
        workspace_mb_override: If set, replaces the profile's
            `workspace_size_mb` for this run only (config.yaml untouched).
            Confirmed on a real Jetson Orin Nano: even `orin_nano_8gb`'s
            1024 MB default can still hit `CUDA error: out of memory` /
            `NvMapMemAllocInternalTagged` at build time if the device's
            free unified memory is lower than usual (desktop session,
            other processes) -- lowering the workspace ceiling makes the
            builder search smaller-memory kernel tactics instead of just
            capping one already-available pool, same idea as week 07's
            `--batch`.

    Returns:
        Path to the written `.engine` file (`fp16_engine_path` if
        `fp16_only`, else `engine_path`).

    Raises:
        FileNotFoundError: If the ONNX model is missing, or (non-`fp16_only`
            only) the calibration cache/manifest is missing.
        RuntimeError: If the ONNX parse or the build itself fails.
    """
    profile = config.get_profile(profile_name)
    if workspace_mb_override is not None:
        profile = dataclasses.replace(profile, workspace_size_mb=workspace_mb_override)

    calibrator = None
    if not fp16_only:
        if not config.paths.calibration_cache_path.is_file():
            raise FileNotFoundError(
                f"Calibration cache not found: {config.paths.calibration_cache_path}. Run "
                "04_quantize_model.py first."
            )
        image_paths = load_calibration_manifest(config.paths.calibration_data_dir)
        # Reuses 04's cache: TensorRT's calibrator checks read_calibration_cache()
        # before ever calling get_batch(), so this calibrator's get_batch() is
        # only a fallback path -- it won't actually run if the cache is present.
        calibrator = build_calibrator(
            image_paths,
            batch_size=profile.batch_size,
            image_size=config.calibration.image_size,
            cache_path=config.paths.calibration_cache_path,
        )

    builder, network, _parser = parse_onnx_network(config.paths.onnx_path)
    builder_config = configure_builder_config(
        builder,
        network,
        bits=None if fp16_only else config.quantization.bits,
        workspace_size_mb=profile.workspace_size_mb,
        fp16_fallback=profile.fp16_fallback,
        batch_size=profile.batch_size,
        calibrator=calibrator,
    )

    serialized_engine = builder.build_serialized_network(network, builder_config)
    if serialized_engine is None:
        raise RuntimeError(
            "TensorRT engine build failed (build_serialized_network returned None) -- check the "
            "builder log above for the underlying error."
        )

    engine_path = config.paths.fp16_engine_path if fp16_only else config.paths.engine_path
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized_engine)
    return engine_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `05_build_engine.py`."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--profile", default=None, help="Builder profile name; defaults to config's active_profile."
    )
    parser.add_argument(
        "--fp16-only",
        action="store_true",
        help="Build the plain-FP16 reference engine (paths.fp16_engine_path) instead of the "
        "quantized one -- no calibration cache needed.",
    )
    parser.add_argument(
        "--workspace-mb",
        type=int,
        default=None,
        help="Override the profile's workspace_size_mb for this run only (config.yaml untouched). "
        "Lower this (e.g. 512 or 256) if the build hits CUDA out-of-memory / "
        "NvMapMemAllocInternalTagged even on the orin_nano_8gb profile's 1024 MB default -- "
        "real Jetson devices can have less free unified memory than usual depending on what "
        "else is running (desktop session, other processes).",
    )
    parser.add_argument(
        "--marker-path",
        default=str(DEFAULT_NV_TEGRA_RELEASE_PATH),
        help="Path checked for Jetson detection. Override only for testing.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip the Jetson-device check (the resulting engine won't be deployable).",
    )
    return parser.parse_args()


def main() -> None:
    """Gate on Jetson, then build and serialize the `.engine` file."""
    args = parse_args()
    require_jetson(Path(args.marker_path), "05_build_engine.py", args.force)

    config = load_config(args.config)
    profile = config.get_profile(args.profile)
    workspace_mb = args.workspace_mb if args.workspace_mb is not None else profile.workspace_size_mb
    bits_label = "FP16-only" if args.fp16_only else f"{config.quantization.bits}-bit"
    print(
        f"Using profile '{args.profile or config.active_profile}': {profile.description}\n"
        f"Building a {bits_label} engine "
        f"(fp16_fallback={profile.fp16_fallback}, workspace={workspace_mb} MB"
        f"{' (overridden)' if args.workspace_mb is not None else ''}, batch={profile.batch_size})."
    )

    engine_path = build_engine(config, args.profile, args.fp16_only, args.workspace_mb)
    print(f"Engine written to '{engine_path}'.")


if __name__ == "__main__":
    main()
