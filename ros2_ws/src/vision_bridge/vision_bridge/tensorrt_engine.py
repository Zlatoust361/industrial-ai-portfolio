"""Minimal TensorRT engine wrapper: load a serialized `.engine`, run inference.

Lazily imports `tensorrt`/`pycuda` -- both are JetPack-provided (or a
matching desktop CUDA+TensorRT install), not pip packages this repo can
install generically, so importing this module never requires either to be
present. Only actually constructing `TensorRTEngine(...)` does, and it
checks the engine file exists *before* that import, so a missing path
fails with a plain `FileNotFoundError` even where `tensorrt`/`pycuda`
aren't installed at all (e.g. this repo's dev sandbox).

Written against the TensorRT 8.5+/10.x Python API (`execute_async_v3` +
`set_tensor_address`, matching current JetPack 6.x). Not runnable in this
sandbox -- no GPU/CUDA/TensorRT here -- so this has been reviewed carefully
against documented API usage but not executed; verify on the actual Jetson
before relying on it (see `course/DECISIONS.md` for the same caveat
pattern on other hardware-specific code this week).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class TensorRTEngineError(RuntimeError):
    """Raised when a `.engine` file can't be deserialized or inference fails.

    The most common cause -- and the reason `detector_factory.build_detector`
    treats this as a fallback trigger, not a hard failure -- is a `.engine`
    built for a different GPU architecture or TensorRT version than the one
    currently running: TensorRT engines are NOT portable across builds,
    only the exact device/TensorRT version they were built on can load them.
    """


class TensorRTEngine:
    """Loads a serialized TensorRT engine and runs inference against it.

    Single fixed input tensor -> one or more output tensors, with
    shapes/dtypes read from the engine's own bindings (never guessed).
    """

    def __init__(self, engine_path: str | Path) -> None:
        """Args:
            engine_path: Path to a serialized `.engine` file.

        Raises:
            FileNotFoundError: If `engine_path` doesn't exist. Checked
                before importing `tensorrt`/`pycuda`, so this is the only
                error this constructor can raise without them installed.
            TensorRTEngineError: If the engine fails to deserialize --
                typically a GPU-architecture/TensorRT-version mismatch (see
                class docstring) -- or `tensorrt`/`pycuda` aren't installed.
        """
        engine_path = Path(engine_path)
        if not engine_path.exists():
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")

        try:
            import pycuda.autoinit  # noqa: F401 -- side effect: initializes a CUDA context
            import pycuda.driver as cuda
            import tensorrt as trt
        except ImportError as exc:
            raise TensorRTEngineError(
                "tensorrt/pycuda are not importable -- they ship with JetPack "
                "(or a matching desktop CUDA+TensorRT install), not pip. "
                f"Original error: {exc}"
            ) from exc

        self._trt = trt
        self._cuda = cuda

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        try:
            with open(engine_path, "rb") as engine_file:
                engine = runtime.deserialize_cuda_engine(engine_file.read())
        except Exception as exc:  # noqa: BLE001 -- tensorrt raises its own opaque errors here
            raise TensorRTEngineError(
                f"Failed to deserialize engine '{engine_path}': {exc}. Most likely "
                "cause: this .engine was built for a different GPU architecture or "
                "TensorRT version -- rebuild it on this exact device (see "
                "ros2_ws/scripts/convert_owlvit_tensorrt.py)."
            ) from exc

        if engine is None:
            raise TensorRTEngineError(
                f"Failed to deserialize engine '{engine_path}' "
                "(deserialize_cuda_engine returned None) -- rebuild it on this exact device."
            )

        self._engine = engine
        self._context = engine.create_execution_context()
        self._stream = cuda.Stream()

        self._input_name = engine.get_tensor_name(0)
        self._output_names = [
            engine.get_tensor_name(i)
            for i in range(engine.num_io_tensors)
            if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT
        ]

    def infer(self, input_array: np.ndarray) -> dict[str, np.ndarray]:
        """Run one inference pass.

        Args:
            input_array: Input tensor, matching the engine's expected input
                shape/dtype exactly (no reshaping/casting is done here).

        Returns:
            Output tensor name -> array, for every output binding.
        """
        cuda = self._cuda
        trt = self._trt
        context = self._context

        input_array = np.ascontiguousarray(input_array)
        context.set_input_shape(self._input_name, input_array.shape)

        device_buffers: dict[str, Any] = {}
        host_outputs: dict[str, np.ndarray] = {}

        input_device = cuda.mem_alloc(input_array.nbytes)
        cuda.memcpy_htod_async(input_device, input_array, self._stream)
        context.set_tensor_address(self._input_name, int(input_device))
        device_buffers[self._input_name] = input_device

        for name in self._output_names:
            shape = context.get_tensor_shape(name)
            dtype = trt.nptype(self._engine.get_tensor_dtype(name))
            host_array = np.empty(tuple(shape), dtype=dtype)
            device_buffer = cuda.mem_alloc(host_array.nbytes)
            context.set_tensor_address(name, int(device_buffer))
            device_buffers[name] = device_buffer
            host_outputs[name] = host_array

        context.execute_async_v3(stream_handle=self._stream.handle)

        for name in self._output_names:
            cuda.memcpy_dtoh_async(host_outputs[name], device_buffers[name], self._stream)
        self._stream.synchronize()

        return host_outputs
