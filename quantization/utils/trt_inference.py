"""TensorRT engine inference for the FP16-vs-INT4 comparison (day 3: week 09).

Shared by `06_trt_inference.py` (latency) and `07_evaluate_quality.py`
(embedding quality) so the actual TensorRT execution/buffer-management
code -- the fiddly, easy-to-get-subtly-wrong part -- exists exactly once.
Lazily imports `tensorrt`/`pycuda`, same reasoning as
`utils/tensorrt_builder.py`: importing this module (e.g. for
`cosine_similarity` in a test) must never require either installed.

Uses TensorRT's "tensor address" execution API (`set_tensor_address`/
`execute_async_v3`, the modern replacement for the older bindings-list
`execute_async_v2`) -- available since TensorRT 8.5, so it's compatible
with JetPack 6.0's bundled TensorRT 8.6 as well as newer TensorRT 10.x
releases, unlike `BuilderFlag.INT4` (see `utils/tensorrt_builder.py`'s
docstring), which TensorRT 8.6 most likely does not have.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from src.evaluation.benchmark import LatencyStats, percentile
from utils.tensorrt_builder import preprocess_calibration_batch

_INPUT_NAME = "pixel_values"
_OUTPUT_NAME = "image_embeds"
"""Match `02_export_onnx.py`'s `torch.onnx.export(..., input_names=[...], output_names=[...])`."""


class TrtImageEncoder:
    """Wraps a deserialized TensorRT engine for batched CLIP image-embedding inference.

    Built for a *fixed* batch size -- the engine itself only has one
    optimization-profile shape (see `utils/tensorrt_builder.py::
    configure_builder_config`), so `encode()` always takes exactly
    `batch_size` images, not a sweep.
    """

    def __init__(self, engine_path: Path, batch_size: int, image_size: int) -> None:
        """Deserialize `engine_path` and allocate its input/output device buffers.

        Args:
            engine_path: A `.engine` file from `05_build_engine.py`.
            batch_size: Must match the batch size the engine was built
                with (its optimization profile's fixed shape).
            image_size: Square edge (pixels) the engine's input expects.

        Raises:
            FileNotFoundError: If `engine_path` doesn't exist.
            RuntimeError: If deserialization fails.
        """
        import numpy as np
        import pycuda.autoinit  # noqa: F401  -- initializes the default CUDA context
        import pycuda.driver as cuda
        import tensorrt as trt

        if not engine_path.is_file():
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}.")

        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)
        engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}.")

        self._batch_size = batch_size
        self._image_size = image_size
        self._cuda = cuda
        self._np = np
        self._engine = engine
        self._context = engine.create_execution_context()

        input_shape = (batch_size, 3, image_size, image_size)
        self._context.set_input_shape(_INPUT_NAME, input_shape)
        output_shape = tuple(self._context.get_tensor_shape(_OUTPUT_NAME))
        self._output_shape = output_shape

        self._device_input = cuda.mem_alloc(int(np.prod(input_shape)) * 4)  # float32
        self._device_output = cuda.mem_alloc(int(np.prod(output_shape)) * 4)
        self._context.set_tensor_address(_INPUT_NAME, int(self._device_input))
        self._context.set_tensor_address(_OUTPUT_NAME, int(self._device_output))
        self._stream = cuda.Stream()

    @property
    def batch_size(self) -> int:
        """The fixed batch size this encoder was built to accept."""
        return self._batch_size

    def encode(self, image_paths: list[Path]) -> Any:
        """Run exactly `self.batch_size` images through the engine.

        Args:
            image_paths: Must have length `self.batch_size`.

        Returns:
            A `(batch_size, embedding_dim)` `float32` NumPy array.

        Raises:
            ValueError: If `len(image_paths) != self.batch_size`.
        """
        if len(image_paths) != self._batch_size:
            raise ValueError(
                f"Expected exactly {self._batch_size} images (this engine's fixed batch size), "
                f"got {len(image_paths)}."
            )
        batch = preprocess_calibration_batch(image_paths, self._image_size)
        self._cuda.memcpy_htod_async(self._device_input, batch.ravel(), self._stream)
        self._context.execute_async_v3(self._stream.handle)
        output = self._np.empty(self._output_shape, dtype=self._np.float32)
        self._cuda.memcpy_dtoh_async(output, self._device_output, self._stream)
        self._stream.synchronize()
        return output


def benchmark_trt_encoder(
    encoder: TrtImageEncoder, image_paths: list[Path], warmup_iters: int, num_iters: int
) -> LatencyStats:
    """Measure `encoder`'s per-batch latency over the fixed test set, cycling as needed.

    Mirrors `src.evaluation.benchmark.benchmark_image_encoder`'s shape
    (reuses its `LatencyStats`/`percentile`), applied to a `TrtImageEncoder`
    instead of a `VLMInference` -- single batch size (the engine's fixed
    one), not a sweep.

    Args:
        encoder: A built `TrtImageEncoder`.
        image_paths: Test images to cycle through (repeated if shorter
            than `encoder.batch_size * (warmup_iters + num_iters)`).
        warmup_iters: Untimed batches run before measurement.
        num_iters: Timed batches.

    Returns:
        A `LatencyStats` for `encoder.batch_size`.

    Raises:
        ValueError: If `image_paths` is empty.
    """
    if not image_paths:
        raise ValueError("image_paths must not be empty.")

    def _batch(index: int) -> list[Path]:
        return [
            image_paths[(index * encoder.batch_size + offset) % len(image_paths)]
            for offset in range(encoder.batch_size)
        ]

    for i in range(warmup_iters):
        encoder.encode(_batch(i))

    durations_ms: list[float] = []
    for i in range(num_iters):
        start = time.perf_counter()
        encoder.encode(_batch(warmup_iters + i))
        durations_ms.append((time.perf_counter() - start) * 1000)
    durations_ms.sort()

    mean_ms = sum(durations_ms) / len(durations_ms)
    throughput = encoder.batch_size / (mean_ms / 1000) if mean_ms > 0 else 0.0
    return LatencyStats(
        batch_size=encoder.batch_size,
        mean_ms=mean_ms,
        p50_ms=percentile(durations_ms, 50),
        p95_ms=percentile(durations_ms, 95),
        throughput_ips=throughput,
    )


def cosine_similarity(a: Any, b: Any) -> Any:
    """Row-wise cosine similarity between two `(N, dim)` embedding arrays.

    Pure NumPy -- no `tensorrt`/`pycuda` needed, so this is unit-testable
    without either installed (unlike `TrtImageEncoder`/`benchmark_trt_encoder`).

    Args:
        a: `(N, dim)` array (e.g. FP16 engine embeddings).
        b: `(N, dim)` array, same shape as `a` (e.g. INT4 engine embeddings).

    Returns:
        `(N,)` array of per-row cosine similarities in `[-1, 1]`.

    Raises:
        ValueError: If `a`/`b` have different shapes.
    """
    import numpy as np

    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}.")
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.sum(a_norm * b_norm, axis=1)
