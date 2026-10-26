"""Shared TensorRT building blocks for `04_quantize_model.py`/`05_build_engine.py`.

Everything here that touches `tensorrt`/`pycuda` imports them lazily, inside
the function/nested class that needs them -- neither package is
pip-installable off a real Jetson/JetPack image (see `../README.md`), so
importing this module itself (e.g. to test `utils/config.py`-adjacent
logic, or the two scripts' argument parsing/Jetson gate) must never require
either installed.

**Honesty note on INT4 specifically**: TensorRT's `IInt8EntropyCalibrator2`
API (used here) has been stable for years and is what `04_quantize_model.py`
relies on for the calibration *cache* regardless of target bit-width.
`trt.BuilderFlag.INT4` (used by `05_build_engine.py` when
`quantization.bits == 4`) is newer and more version/hardware-gated than the
long-established `INT8` flag -- most likely newer than JetPack 6.0's
bundled TensorRT 8.6 itself (INT4 weight-only quantization support landed
around TensorRT 10.x) -- `configure_builder_config` checks it exists on
the installed `tensorrt` build before using it and raises a clear error
if not, rather than silently falling back to something the config didn't
ask for. None of this has been run against real TensorRT/a real Jetson in
this sandbox (`tensorrt` isn't installed here) -- see
`course/DECISIONS.md`'s week 09 day 2/3 entries.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# transformers.CLIPImageProcessor's default normalization constants -- the
# calibration batches must match the same input distribution the exported
# ONNX graph (02_export_onnx.py) actually expects, not an arbitrary [0, 1]
# scale. Assumes a CLIP-family model (config.model.type == "clip"), the
# only family this repo has ever loaded.
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

_ONNX_INPUT_NAME = "pixel_values"
"""Matches `02_export_onnx.py`'s `torch.onnx.export(..., input_names=["pixel_values"])`."""


def load_calibration_manifest(calibration_data_dir: Path) -> list[Path]:
    """Read `manifest.json` written by `03_prepare_calibration_data.py`.

    Args:
        calibration_data_dir: Directory `03` wrote its images + manifest to.

    Returns:
        Preprocessed calibration image paths, in manifest order.

    Raises:
        FileNotFoundError: If the manifest doesn't exist -- run
            `03_prepare_calibration_data.py` first.
    """
    manifest_path = calibration_data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Calibration manifest not found: {manifest_path}. Run "
            "03_prepare_calibration_data.py first."
        )
    manifest = json.loads(manifest_path.read_text())
    return [Path(entry["output"]) for entry in manifest["images"]]


def preprocess_calibration_batch(image_paths: list[Path], image_size: int) -> Any:
    """Load and normalize one batch of calibration images to CLIP's input distribution.

    Args:
        image_paths: Paths to preprocessed (already-resized) calibration images.
        image_size: Expected square edge (pixels); must match how `03` saved them.

    Returns:
        A `(len(image_paths), 3, image_size, image_size)` `float32` NumPy
        array, channel-normalized with `_CLIP_MEAN`/`_CLIP_STD`.
    """
    import numpy as np
    from PIL import Image

    batch = np.zeros((len(image_paths), 3, image_size, image_size), dtype=np.float32)
    for index, path in enumerate(image_paths):
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        for channel in range(3):
            image[:, :, channel] = (
                image[:, :, channel] - _CLIP_MEAN[channel]
            ) / _CLIP_STD[channel]
        batch[index] = image.transpose(2, 0, 1)
    return batch


def build_calibrator(
    image_paths: list[Path], batch_size: int, image_size: int, cache_path: Path
) -> Any:
    """Build a TensorRT `IInt8EntropyCalibrator2` feeding preprocessed calibration images.

    Standard TensorRT Python calibrator shape: `get_batch_size`/`get_batch`
    are called repeatedly by the builder while it runs calibration,
    `get_batch` returning `None` signals "no more batches".
    `read_calibration_cache`/`write_calibration_cache` let a
    previously-computed cache short-circuit recalibration on a later build
    -- `05_build_engine.py` reuses the cache `04_quantize_model.py` wrote.

    Args:
        image_paths: Preprocessed calibration image paths (see `load_calibration_manifest`).
        batch_size: Images per calibration batch (from the active builder profile).
        image_size: Square edge (pixels) each image was resized to.
        cache_path: Where to read/write the calibration cache.

    Returns:
        An instance ready to assign to `tensorrt.IBuilderConfig.int8_calibrator`.
    """
    import pycuda.autoinit  # noqa: F401  -- initializes the default CUDA context
    import pycuda.driver as cuda
    import tensorrt as trt

    class ImageEntropyCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self) -> None:
            super().__init__()
            self._image_paths = image_paths
            self._batch_size = batch_size
            self._image_size = image_size
            self._cache_path = cache_path
            self._index = 0
            self._device_input = cuda.mem_alloc(batch_size * 3 * image_size * image_size * 4)

        def get_batch_size(self) -> int:
            return self._batch_size

        def get_batch(self, names: list[str]) -> list[int] | None:
            if self._index + self._batch_size > len(self._image_paths):
                return None
            batch_paths = self._image_paths[self._index : self._index + self._batch_size]
            batch = preprocess_calibration_batch(batch_paths, self._image_size)
            cuda.memcpy_htod(self._device_input, batch.ravel())
            self._index += self._batch_size
            return [int(self._device_input)]

        def read_calibration_cache(self) -> bytes | None:
            return self._cache_path.read_bytes() if self._cache_path.is_file() else None

        def write_calibration_cache(self, cache: bytes) -> None:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_bytes(cache)

    return ImageEntropyCalibrator()


def parse_onnx_network(onnx_path: Path) -> tuple[Any, Any, Any]:
    """Create a TensorRT network (explicit batch) and parse `onnx_path` into it.

    Args:
        onnx_path: Path to the ONNX graph exported by `02_export_onnx.py`.

    Returns:
        `(builder, network, parser)` -- all three kept alive by the caller
        for the rest of the build (TensorRT's Python bindings don't keep
        their own references).

    Raises:
        FileNotFoundError: If `onnx_path` doesn't exist.
        RuntimeError: If the ONNX parser reports errors.
    """
    import tensorrt as trt

    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}. Run 02_export_onnx.py first.")

    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, trt_logger)

    if not parser.parse(onnx_path.read_bytes()):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"Failed to parse ONNX model '{onnx_path}':\n{errors}")

    return builder, network, parser


def configure_builder_config(
    builder: Any,
    network: Any,
    bits: int | None,
    workspace_size_mb: int,
    fp16_fallback: bool,
    batch_size: int,
    calibrator: Any | None,
) -> Any:
    """Build and configure a TensorRT `IBuilderConfig` from this week's profile settings.

    Args:
        builder: From `parse_onnx_network`.
        network: From `parse_onnx_network` -- read for its (dynamic-batch) input shape.
        bits: `4` or `8` for a quantized build (`quantization.bits`), or
            `None` for a plain FP16-only build -- the reference engine
            `06_trt_inference.py`/`07_evaluate_quality.py` compare INT4
            against. `None` always sets `trt.BuilderFlag.FP16` regardless
            of `fp16_fallback` (there's no "fallback" concept without a
            lower-bit target) and needs no `calibrator`.
        workspace_size_mb: Builder scratch memory for kernel autotuning, in MiB.
        fp16_fallback: Also set `trt.BuilderFlag.FP16` when `bits` is not
            `None`, so layers without an INT4/INT8 kernel fall back to
            fp16 instead of failing the build. Ignored when `bits is None`
            (FP16 is set unconditionally in that case).
        batch_size: Fixed engine input batch size -- an `IOptimizationProfile`
            is added pinning min=opt=max to this value (see the profile's
            `description` in `config.yaml` for why a fixed, not dynamic, batch).
        calibrator: From `build_calibrator` -- required when `bits` is `4`
            or `8` (TensorRT's INT4 weight-only path still consumes an
            `IInt8Calibrator` for activation ranges in the TensorRT
            versions this was written against; see this module's docstring
            for how unverified that specific claim is). Ignored when `bits is None`.

    Returns:
        A configured `tensorrt.IBuilderConfig`, ready for `builder.build_serialized_network`.

    Raises:
        ValueError: If `bits` isn't `None`, `4`, or `8`.
        RuntimeError: If `bits == 4` but the installed `tensorrt` has no
            `BuilderFlag.INT4` (older TensorRT builds don't) -- fails loudly
            instead of silently building an INT8 or fp16 engine the config
            didn't ask for.
    """
    import tensorrt as trt

    if bits not in (None, 4, 8):
        raise ValueError(f"Unsupported bits={bits!r}; expected None, 4, or 8.")

    builder_config = builder.create_builder_config()
    builder_config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, workspace_size_mb * (1024**2)
    )

    if bits is None:
        builder_config.set_flag(trt.BuilderFlag.FP16)
    else:
        if fp16_fallback:
            builder_config.set_flag(trt.BuilderFlag.FP16)
        if bits == 8:
            builder_config.set_flag(trt.BuilderFlag.INT8)
        else:
            int4_flag = getattr(trt.BuilderFlag, "INT4", None)
            if int4_flag is None:
                raise RuntimeError(
                    "quantization.bits=4 was requested, but this installed tensorrt build has "
                    "no BuilderFlag.INT4 -- INT4 weight-only quantization support is newer and "
                    "more version-gated than INT8 (see utils/tensorrt_builder.py's module "
                    "docstring). Upgrade TensorRT/JetPack, or set quantization.bits: 8 in "
                    "config.yaml."
                )
            builder_config.set_flag(int4_flag)
        builder_config.int8_calibrator = calibrator

    optimization_profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)
    fixed_shape = (batch_size, *input_tensor.shape[1:])
    optimization_profile.set_shape(_ONNX_INPUT_NAME, fixed_shape, fixed_shape, fixed_shape)
    builder_config.add_optimization_profile(optimization_profile)

    return builder_config
