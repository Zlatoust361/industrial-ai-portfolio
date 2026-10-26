"""Typed loader for `config/config.yaml` (week 09: INT4 quantization + TensorRT).

Mirrors `src/config.py`'s pattern from week 01 and `pipeline/utils/config.py`'s
from week 04: one frozen dataclass per YAML section, `load_config()` parses
and validates. A separate schema, not reused from either -- this week's
config is about a quantization/TensorRT build recipe (ONNX export,
calibration data, builder profiles), nothing in common with week 01's
single-checkpoint inference config or week 04's 7-stage pipeline.

`profiles:` is the one new shape relative to those two: a *named* dict of
builder-sizing presets (`dev`, `orin_nano_8gb`, ...) rather than a fixed
two-value `dev|jetson` switch -- `get_profile()` looks one up by name,
falling back to `active_profile` from the YAML.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

QUANTIZATION_METHODS: tuple[str, ...] = ("ptq",)
"""Valid `quantization.method` values -- only post-training quantization, for now."""

QUANTIZATION_BITS: tuple[int, ...] = (4, 8)
"""Valid `quantization.bits` values TensorRT's builder is asked to target."""


@dataclass(frozen=True)
class ModelConfig:
    """`model:` section -- which checkpoint the whole recipe starts from.

    Attributes:
        name: HuggingFace hub identifier or local path of the checkpoint.
        type: Model family, e.g. `"clip"`.
        precision: `"fp16"` or `"fp32"` -- the week's own baseline
            (`01_baseline_inference.py`/`02_export_onnx.py`) starting point.
        device: `"cpu"` or `"cuda"`, used by 01/02 only -- the engine
            build's own device comes from the active profile, not this field.
        seed: Seed applied before model construction, and before sampling
            calibration images (`03_prepare_calibration_data.py`).
        cache_dir: Local directory used to cache downloaded weights.
    """

    name: str
    type: str
    precision: str
    device: str
    seed: int
    cache_dir: Path


@dataclass(frozen=True)
class MemoryConfig:
    """`memory:` section -- same edge-device guardrail every config in this repo uses.

    Attributes:
        memory_budget_mb: Passed straight to `src.models.memory.check_memory_budget`/
            `check_process_memory` via `CLIPLoader`. Default 6144 MB (ADR-002).
        device_total_mb: Total device RAM, for a clearer over-budget message.
    """

    memory_budget_mb: int
    device_total_mb: int


@dataclass(frozen=True)
class PathsConfig:
    """`paths:` section -- where each script reads/writes, relative to the invocation cwd.

    Attributes:
        outputs_dir: Root of every artifact below.
        onnx_path: Exported ONNX image encoder (`02_export_onnx.py`).
        calibration_data_dir: Preprocessed calibration images (`03_prepare_calibration_data.py`).
        calibration_cache_path: TensorRT calibration cache (`04_quantize_model.py`).
        engine_path: Serialized, quantized `.engine` file (`05_build_engine.py`).
        fp16_engine_path: Serialized, plain-FP16 `.engine` file
            (`05_build_engine.py --fp16-only`) -- the reference engine
            `06_trt_inference.py`/`07_evaluate_quality.py` compare
            `engine_path`'s INT4/INT8 build against.
    """

    outputs_dir: Path
    onnx_path: Path
    calibration_data_dir: Path
    calibration_cache_path: Path
    engine_path: Path
    fp16_engine_path: Path


@dataclass(frozen=True)
class BaselineConfig:
    """`baseline:` section -- `01_baseline_inference.py`'s benchmark sweep.

    Attributes:
        image_size: Square edge (pixels) of the synthetic/sample benchmark image.
        batch_sizes: Batch sizes to sweep (see `src.evaluation.benchmark.benchmark_image_encoder`).
        warmup_iters: Untimed iterations run before measurement, per batch size.
        num_iters: Timed iterations per batch size.
    """

    image_size: int
    batch_sizes: list[int]
    warmup_iters: int
    num_iters: int


@dataclass(frozen=True)
class OnnxConfig:
    """`onnx:` section -- `02_export_onnx.py`'s export + smoke-test settings.

    Attributes:
        opset: ONNX opset version used for export.
        dynamic_batch: Whether to mark the batch axis dynamic on the
            exported graph (the trace itself is always batch=1 -- the Orin
            Nano's single-video-stream scenario).
        atol: Absolute tolerance for the PyTorch-vs-ONNXRuntime smoke test.
        rtol: Relative tolerance for the PyTorch-vs-ONNXRuntime smoke test.
    """

    opset: int
    dynamic_batch: bool
    atol: float
    rtol: float


@dataclass(frozen=True)
class CalibrationConfig:
    """`calibration:` section -- `03_prepare_calibration_data.py`'s image sampling.

    Attributes:
        source_dir: Directory of representative images to sample from
            (week 06's stratified val subset by default).
        num_samples: How many images to sample for calibration.
        image_size: Square edge (pixels) calibration images are resized to.
    """

    source_dir: Path
    num_samples: int
    image_size: int


@dataclass(frozen=True)
class EvaluationConfig:
    """`evaluation:` section -- `06_trt_inference.py`/`07_evaluate_quality.py`'s fixed test set.

    Attributes:
        test_set_dir: Week 06's held-out test split (distinct from
            `calibration.source_dir`'s val split -- calibrating and
            evaluating on the same images would be optimistic).
        num_samples: Cap on how many test images each comparison run uses
            -- keeps a run bounded; `None` in `06`/`07`'s `--num-samples`
            CLI override means "every discovered image".
        warmup_iters: Untimed iterations run before latency measurement.
        num_iters: Timed iterations per engine (`06_trt_inference.py`).
        min_cosine_similarity: `07_evaluate_quality.py`'s quality gate --
            the INT4 engine's embeddings must stay at or above this mean
            cosine similarity to the FP16 engine's on the same images, or
            the script exits `1` (a real gate, like week 04/06's
            `06_quality_gate.py`/`08_validate_dataset.py`).
        throttle_temp_c: `08_profile.py`'s thermal-warning threshold, in
            Celsius -- same default (85.0) and same "configurable proxy,
            not a real DVFS trip point" caveat as week 08's
            `src/api/metrics.py::THROTTLE_TEMP_C`.
    """

    test_set_dir: Path
    num_samples: int
    warmup_iters: int
    num_iters: int
    min_cosine_similarity: float
    throttle_temp_c: float


@dataclass(frozen=True)
class QuantizationSettings:
    """`quantization:` section -- what to quantize to, and how.

    Attributes:
        method: One of `QUANTIZATION_METHODS`.
        bits: One of `QUANTIZATION_BITS` -- the target weight bit-width.
        calibrator: TensorRT calibrator algorithm name (see `04_quantize_model.py`).
        per_channel: Per-output-channel scales (`True`) vs. one scale for
            the whole tensor (`False`).
    """

    method: str
    bits: int
    calibrator: str
    per_channel: bool


@dataclass(frozen=True)
class BuilderProfile:
    """One named entry under `profiles:` -- TensorRT builder sizing for one target.

    Attributes:
        description: Human-readable rationale, printed by `04`/`05` at
            startup so whoever runs them sees which profile (and why) is
            in effect, not just its name.
        device: `"cuda"` (TensorRT always targets a CUDA device; kept as an
            explicit field rather than assumed, for a clearer error if
            something's misconfigured).
        batch_size: Fixed engine input batch size (see `orin_nano_8gb`'s
            `description` in `config.yaml` for why 1, not dynamic, on-device).
        workspace_size_mb: TensorRT builder scratch memory for kernel
            autotuning, in MiB (`trt.MemoryPoolType.WORKSPACE`).
        fp16_fallback: Whether to also set `trt.BuilderFlag.FP16` alongside
            the low-bit flag, letting the builder fall back to fp16 for
            layers without an INT4/INT8 kernel instead of failing the build.
    """

    description: str
    device: str
    batch_size: int
    workspace_size_mb: int
    fp16_fallback: bool


@dataclass(frozen=True)
class QuantizationConfig:
    """Root config object composed of every section in `config.yaml`."""

    model: ModelConfig
    memory: MemoryConfig
    paths: PathsConfig
    baseline: BaselineConfig
    onnx: OnnxConfig
    calibration: CalibrationConfig
    evaluation: EvaluationConfig
    quantization: QuantizationSettings
    active_profile: str
    profiles: dict[str, BuilderProfile]

    def get_profile(self, name: str | None = None) -> BuilderProfile:
        """Look up a named builder profile, defaulting to `active_profile`.

        Args:
            name: Profile name (e.g. `"orin_nano_8gb"`), or `None` to use
                `self.active_profile`.

        Returns:
            The matching `BuilderProfile`.

        Raises:
            KeyError: If the resolved name isn't in `self.profiles`.
        """
        profile_name = name or self.active_profile
        if profile_name not in self.profiles:
            raise KeyError(
                f"Unknown profile {profile_name!r}; expected one of "
                f"{sorted(self.profiles)} (see config.yaml's profiles: section)."
            )
        return self.profiles[profile_name]


def load_config(path: str | Path = "config/config.yaml") -> QuantizationConfig:
    """Load and validate the quantization config from a YAML file.

    Args:
        path: Path to a YAML file matching `config/config.yaml`'s schema.

    Returns:
        A fully parsed, immutable `QuantizationConfig`.

    Raises:
        FileNotFoundError: If `path` does not exist.
        KeyError: If a required section or field is missing from the YAML.
        ValueError: If `quantization.method`/`quantization.bits` isn't a
            recognized value, or `active_profile` isn't a key of `profiles`.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Quantization config file not found: {config_path}")

    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    model_raw = raw["model"]
    memory_raw = raw["memory"]
    paths_raw = raw["paths"]
    baseline_raw = raw["baseline"]
    onnx_raw = raw["onnx"]
    calibration_raw = raw["calibration"]
    evaluation_raw = raw["evaluation"]
    quantization_raw = raw["quantization"]
    profiles_raw = raw["profiles"]
    active_profile = raw["active_profile"]

    if quantization_raw["method"] not in QUANTIZATION_METHODS:
        raise ValueError(
            f"Unknown quantization.method {quantization_raw['method']!r}; "
            f"expected one of {QUANTIZATION_METHODS}."
        )
    if int(quantization_raw["bits"]) not in QUANTIZATION_BITS:
        raise ValueError(
            f"Unknown quantization.bits {quantization_raw['bits']!r}; "
            f"expected one of {QUANTIZATION_BITS}."
        )
    if active_profile not in profiles_raw:
        raise ValueError(
            f"active_profile {active_profile!r} is not a key of profiles: "
            f"{sorted(profiles_raw)}."
        )

    return QuantizationConfig(
        model=ModelConfig(
            name=model_raw["name"],
            type=model_raw["type"],
            precision=model_raw["precision"],
            device=model_raw["device"],
            seed=int(model_raw["seed"]),
            cache_dir=Path(model_raw["cache_dir"]),
        ),
        memory=MemoryConfig(
            memory_budget_mb=int(memory_raw["memory_budget_mb"]),
            device_total_mb=int(memory_raw["device_total_mb"]),
        ),
        paths=PathsConfig(
            outputs_dir=Path(paths_raw["outputs_dir"]),
            onnx_path=Path(paths_raw["onnx_path"]),
            calibration_data_dir=Path(paths_raw["calibration_data_dir"]),
            calibration_cache_path=Path(paths_raw["calibration_cache_path"]),
            engine_path=Path(paths_raw["engine_path"]),
            fp16_engine_path=Path(paths_raw["fp16_engine_path"]),
        ),
        baseline=BaselineConfig(
            image_size=int(baseline_raw["image_size"]),
            batch_sizes=list(baseline_raw["batch_sizes"]),
            warmup_iters=int(baseline_raw["warmup_iters"]),
            num_iters=int(baseline_raw["num_iters"]),
        ),
        onnx=OnnxConfig(
            opset=int(onnx_raw["opset"]),
            dynamic_batch=bool(onnx_raw["dynamic_batch"]),
            atol=float(onnx_raw["atol"]),
            rtol=float(onnx_raw["rtol"]),
        ),
        calibration=CalibrationConfig(
            source_dir=Path(calibration_raw["source_dir"]),
            num_samples=int(calibration_raw["num_samples"]),
            image_size=int(calibration_raw["image_size"]),
        ),
        evaluation=EvaluationConfig(
            test_set_dir=Path(evaluation_raw["test_set_dir"]),
            num_samples=int(evaluation_raw["num_samples"]),
            warmup_iters=int(evaluation_raw["warmup_iters"]),
            num_iters=int(evaluation_raw["num_iters"]),
            min_cosine_similarity=float(evaluation_raw["min_cosine_similarity"]),
            throttle_temp_c=float(evaluation_raw["throttle_temp_c"]),
        ),
        quantization=QuantizationSettings(
            method=quantization_raw["method"],
            bits=int(quantization_raw["bits"]),
            calibrator=quantization_raw["calibrator"],
            per_channel=bool(quantization_raw["per_channel"]),
        ),
        active_profile=active_profile,
        profiles={
            name: BuilderProfile(
                description=str(profile_raw["description"]).strip(),
                device=profile_raw["device"],
                batch_size=int(profile_raw["batch_size"]),
                workspace_size_mb=int(profile_raw["workspace_size_mb"]),
                fp16_fallback=bool(profile_raw["fp16_fallback"]),
            )
            for name, profile_raw in profiles_raw.items()
        },
    )
