"""Typed loader for `config/model_config.yaml`.

Every runtime parameter (model name, precision, device, export/benchmark
settings, ...) lives in YAML and is parsed into immutable dataclasses here,
so no module in `src/` hardcodes model names or paths.

Package, not a single file, as of week 12 (capstone) -- every name below
was previously `src/config.py`'s module-level content, moved into
`__init__.py` unchanged (same public API, same `from src.config import
AppConfig, load_config` call sites across `src/`/`scripts/`/`pipeline/`/
`quantization/`, none of them needed to change). `src/config/environment.py`
adds the one genuinely new piece this week needs: a single dev/jetson
environment switch shared by every capstone layer (`vision`/`nlp`/
`robot`/`database`/`scenarios`), instead of each one re-deriving it from
`TARGETS`/`apply_target_overrides` independently.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

TARGETS: tuple[str, ...] = ("dev", "jetson")
"""Valid `--target` values accepted by `apply_target_overrides` and the CLI scripts."""


@dataclass(frozen=True)
class ModelSection:
    """`model:` section of the config.

    Attributes:
        name: HuggingFace hub identifier or local path of the checkpoint.
        type: Model family, e.g. `"clip"`.
        precision: `"fp16"` or `"fp32"`.
        device: `"cpu"` or `"cuda"`.
        seed: Seed applied before model construction for reproducibility.
        cache_dir: Local directory used to cache downloaded weights.
        backend: Which `VLMInference` implementation `src.models.factory.build_inference`
            builds -- `"pytorch"` (default; the plain HuggingFace `CLIPLoader`/
            `CLIPInference` path, unchanged from before this field existed),
            `"tensorrt"` (require a working `engine_path`, raise if unavailable),
            or `"auto"` (try `"tensorrt"` if `engine_path` is set and the file
            exists, fall back to `"pytorch"` on any load failure -- mirrors
            `ros2_ws/src/vision_bridge/vision_bridge/detector_factory.py`'s
            `backend="auto"` semantics for OWL-ViT detection).
        engine_path: Path to a TensorRT `.engine` file built by week 9's
            `quantization/scripts/05_build_engine.py` (the INT4/INT8-quantized
            image encoder). `None`/unset means no engine is configured --
            `backend="auto"` then always falls back to PyTorch, and
            `backend="tensorrt"` raises. Ignored when `backend="pytorch"`.
    """

    name: str
    type: str
    precision: str
    device: str
    seed: int
    cache_dir: Path
    backend: str = "pytorch"
    engine_path: Path | None = None


@dataclass(frozen=True)
class InferenceSection:
    """`inference:` section of the config.

    Attributes:
        batch_size: Default batch size used by inference callers.
        image_size: Square edge (pixels) images are resized to before encoding.
    """

    batch_size: int
    image_size: int


@dataclass(frozen=True)
class ApiSection:
    """`api:` section of the config.

    Attributes:
        host: Bind host for the FastAPI server.
        port: Bind port for the FastAPI server.
    """

    host: str
    port: int


@dataclass(frozen=True)
class ExportSection:
    """`export:` section of the config.

    Attributes:
        onnx_opset: ONNX opset version used for export.
        onnx_output_path: Destination path for the exported `.onnx` file.
        dynamic_batch: Whether to mark the batch axis dynamic on the exported
            graph, so it accepts any batch size at inference time. Export
            itself is always traced at batch=1 (see `scripts/02_convert_onnx.py`)
            since a single edge video stream has nothing to batch.
        onnx_atol: Default absolute tolerance for the PyTorch-vs-ONNXRuntime
            smoke test (`numpy.allclose`); overridable via `--atol`.
        onnx_rtol: Default relative tolerance for the PyTorch-vs-ONNXRuntime
            smoke test (`numpy.allclose`); overridable via `--rtol`.
    """

    onnx_opset: int
    onnx_output_path: Path
    dynamic_batch: bool
    onnx_atol: float
    onnx_rtol: float


@dataclass(frozen=True)
class MemorySection:
    """`memory:` section of the config -- edge-device memory guardrails.

    Attributes:
        memory_budget_mb: Maximum RAM the loaded model + inference runtime
            (activations, CUDA/TensorRT context, ...) may use, in MiB.
            Defaults to 6144 MB on an 8192 MB Jetson Orin Nano, leaving
            ~2 GB of headroom for the OS and other processes.
        device_total_mb: Total device RAM, in MiB. Used only to produce a
            clearer error message when the budget is exceeded.
    """

    memory_budget_mb: int
    device_total_mb: int


@dataclass(frozen=True)
class BenchmarkSection:
    """`benchmark:` section of the config.

    Attributes:
        warmup_iters: Untimed iterations run before measurement.
        num_iters: Timed iterations per batch size.
        batch_sizes: Batch sizes to sweep during benchmarking.
        output_dir: Directory the benchmark report is written to.
    """

    warmup_iters: int
    num_iters: int
    batch_sizes: list[int]
    output_dir: Path


@dataclass(frozen=True)
class AppConfig:
    """Root config object composed of every section in `model_config.yaml`."""

    model: ModelSection
    inference: InferenceSection
    api: ApiSection
    export: ExportSection
    memory: MemorySection
    benchmark: BenchmarkSection


def load_config(path: str | Path = "config/model_config.yaml") -> AppConfig:
    """Load and validate the application config from a YAML file.

    Args:
        path: Path to a YAML file matching the `config/model_config.yaml` schema.

    Returns:
        A fully parsed, immutable `AppConfig` instance.

    Raises:
        FileNotFoundError: If `path` does not exist.
        KeyError: If a required section or field is missing from the YAML.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    model_raw = raw["model"]
    inference_raw = raw["inference"]
    api_raw = raw["api"]
    export_raw = raw["export"]
    memory_raw = raw["memory"]
    benchmark_raw = raw["benchmark"]

    engine_path_raw = model_raw.get("engine_path")

    return AppConfig(
        model=ModelSection(
            name=model_raw["name"],
            type=model_raw["type"],
            precision=model_raw["precision"],
            device=model_raw["device"],
            seed=int(model_raw["seed"]),
            cache_dir=Path(model_raw["cache_dir"]),
            backend=model_raw.get("backend", "pytorch"),
            engine_path=Path(engine_path_raw) if engine_path_raw else None,
        ),
        inference=InferenceSection(
            batch_size=int(inference_raw["batch_size"]),
            image_size=int(inference_raw["image_size"]),
        ),
        api=ApiSection(
            host=api_raw["host"],
            port=int(api_raw["port"]),
        ),
        export=ExportSection(
            onnx_opset=int(export_raw["onnx_opset"]),
            onnx_output_path=Path(export_raw["onnx_output_path"]),
            dynamic_batch=bool(export_raw["dynamic_batch"]),
            onnx_atol=float(export_raw["onnx_atol"]),
            onnx_rtol=float(export_raw["onnx_rtol"]),
        ),
        memory=MemorySection(
            memory_budget_mb=int(memory_raw["memory_budget_mb"]),
            device_total_mb=int(memory_raw["device_total_mb"]),
        ),
        benchmark=BenchmarkSection(
            warmup_iters=int(benchmark_raw["warmup_iters"]),
            num_iters=int(benchmark_raw["num_iters"]),
            batch_sizes=list(benchmark_raw["batch_sizes"]),
            output_dir=Path(benchmark_raw["output_dir"]),
        ),
    )


def apply_target_overrides(config: AppConfig, target: str) -> AppConfig:
    """Apply `--target dev|jetson` overrides on top of a loaded config.

    `target="jetson"` forces fp16 precision, since the on-device Jetson Orin
    Nano deployment path always runs fp16. `target="dev"` is a no-op, for
    running/benchmarking on a regular development machine.

    Args:
        config: A loaded `AppConfig`.
        target: One of `TARGETS` (`"dev"` or `"jetson"`).

    Returns:
        `config` unchanged for `"dev"`; a copy with `model.precision` forced
        to `"fp16"` for `"jetson"`.

    Raises:
        ValueError: If `target` is not one of `TARGETS`.
    """
    if target not in TARGETS:
        raise ValueError(f"Unknown target {target!r}; expected one of {TARGETS}.")
    if target == "dev":
        return config
    return replace(config, model=replace(config.model, precision="fp16"))
