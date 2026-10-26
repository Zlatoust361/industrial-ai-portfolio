"""Typed loader for `config/pipeline_config.yaml`.

Every one of the 7 stage scripts (`scripts/01_environment_check.py` ...
`scripts/07_generate_report.py`) reads its paths/thresholds/model names
through `load_config()` here -- none of them hardcode a value the YAML
already owns. Mirrors `src/config.py`'s pattern from week 01
(`load_config` + an explicit `apply_*_overrides` for the dev/jetson
switch), kept as a separate module (not a reuse of `src/config.py`)
because this week's schema (7 pipeline stages, quality-gate thresholds,
report title, ...) has nothing in common with week 01's CLIP-benchmark
config beyond the general "typed YAML section" shape.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

DEVICE_TARGETS: tuple[str, ...] = ("dev", "jetson")
"""Valid `device.target` / `--device` values, same two-value shape as week 01's `TARGETS`."""

DETECT_BACKENDS: tuple[str, ...] = ("clip", "yolo", "stub")
"""Valid `detect.backend` values (see `scripts/04_detect.py`). `stub` needs
no torch/transformers/ultralytics -- it's what `run_all.sh`'s demo run and
CI's quality gate use, see `04_detect.py::StubDetector`."""


@dataclass(frozen=True)
class PathsConfig:
    """`paths:` section -- where each stage reads/writes, relative to the invocation cwd.

    Attributes:
        data_dir: Root of stage-to-stage data artifacts (raw/processed images, manifests).
        outputs_dir: Root of pipeline results (detections, metrics, the final report).
        logs_dir: Root of per-stage log files (wired up day 3's run_all.sh).
    """

    data_dir: Path
    outputs_dir: Path
    logs_dir: Path


@dataclass(frozen=True)
class DeviceConfig:
    """`device:` section -- the dev/Jetson switch every stage reads.

    Attributes:
        target: `"dev"` or `"jetson"`. Not auto-detected here on purpose --
            `01_environment_check.py` reports what device is actually
            running the pipeline, but a human/CI decides `target`
            explicitly, so a dev machine can still dry-run the jetson
            profile (fp16, batch=1) without pretending to be one.
        precision: `"fp32"` or `"fp16"`; forced to `"fp16"` by
            `apply_device_target(config, "jetson")`.
        batch_size: Forced to `1` by `apply_device_target(config, "jetson")`
            -- the Orin Nano reads one video stream, nothing to batch.
        torch_device: `"cpu"` or `"cuda"`, passed straight through to
            `src.config.ModelSection.device` when `04_detect.py`'s `clip`
            backend builds a `CLIPLoader`.
        seed: Deterministic seed applied before model construction.
        memory_budget_mb: Same guardrail as week 01's `MemorySection`
            (default 6144 MB on an 8192 MB Orin Nano).
        device_total_mb: Total device RAM, for a clearer over-budget message.
    """

    target: str
    precision: str
    batch_size: int
    torch_device: str
    seed: int
    memory_budget_mb: int
    device_total_mb: int


@dataclass(frozen=True)
class IngestConfig:
    """`ingest:` section.

    Attributes:
        input_glob: Filename glob resolved against `<data_dir>/raw/`.
    """

    input_glob: str


@dataclass(frozen=True)
class PreprocessConfig:
    """`preprocess:` section.

    Attributes:
        image_size: `(width, height)` each raw image is resized to before
            being saved under `<data_dir>/processed/`.
    """

    image_size: tuple[int, int]


@dataclass(frozen=True)
class DetectConfig:
    """`detect:` section.

    Attributes:
        backend: `"clip"` (zero-shot ok/defect via `src.models.zero_shot`,
            reusing week 02's tested classification code), `"yolo"`
            (`ultralytics.YOLO`), or `"stub"` (dependency-light, deterministic
            pixel-marker check -- no torch/transformers/ultralytics; used by
            `run_all.sh`'s demo run and CI). See `scripts/04_detect.py`.
        clip_model_name: HuggingFace checkpoint for the `clip` backend.
        clip_cache_dir: Local weights cache for the `clip` backend.
        clip_prompt_style: Style name (`basic`/`industrial`/`detailed`) from `clip_prompts_path`.
        clip_prompts_path: Path to a `PromptManager`-compatible YAML file
            (week 02's `config/prompts.yaml` by default).
        yolo_model_path: Checkpoint path/name for the `yolo` backend.
        yolo_defect_classes: YOLO class names that count as a defect;
            empty means "every detected box counts as a defect" (see
            `scripts/04_detect.py`'s yolo backend docstring).
        score_threshold: Minimum confidence a detection needs to be kept
            at all (applied inside the backend, before `05_interpret.py`'s
            separate ok/defect threshold).
    """

    backend: str
    clip_model_name: str
    clip_cache_dir: Path
    clip_prompt_style: str
    clip_prompts_path: Path
    yolo_model_path: str
    yolo_defect_classes: tuple[str, ...]
    score_threshold: float


@dataclass(frozen=True)
class InterpretConfig:
    """`interpret:` section.

    Attributes:
        defect_score_threshold: A detection scoring at or above this on
            the "defect" class/prompt makes the image's verdict "defect".
    """

    defect_score_threshold: float


@dataclass(frozen=True)
class QualityGateConfig:
    """`quality_gate:` section.

    Attributes:
        green_max_defect_rate: `defect_rate` strictly below this -> GREEN.
        yellow_max_defect_rate: `defect_rate` strictly below this (and not
            GREEN) -> YELLOW; at or above it -> RED.
    """

    green_max_defect_rate: float
    yellow_max_defect_rate: float


@dataclass(frozen=True)
class ReportConfig:
    """`report:` section.

    Attributes:
        title: Heading used at the top of the generated Markdown report.
    """

    title: str


@dataclass(frozen=True)
class PipelineConfig:
    """Root config object composed of every section in `pipeline_config.yaml`."""

    paths: PathsConfig
    device: DeviceConfig
    ingest: IngestConfig
    preprocess: PreprocessConfig
    detect: DetectConfig
    interpret: InterpretConfig
    quality_gate: QualityGateConfig
    report: ReportConfig


def load_config(path: str | Path = "config/pipeline_config.yaml") -> PipelineConfig:
    """Load and validate the pipeline config from a YAML file.

    Args:
        path: Path to a YAML file matching `pipeline_config.yaml`'s schema.

    Returns:
        A fully parsed, immutable `PipelineConfig`.

    Raises:
        FileNotFoundError: If `path` does not exist.
        KeyError: If a required section or field is missing from the YAML.
        ValueError: If `device.target` or `detect.backend` isn't a recognized value.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Pipeline config file not found: {config_path}")

    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    paths_raw = raw["paths"]
    device_raw = raw["device"]
    ingest_raw = raw["ingest"]
    preprocess_raw = raw["preprocess"]
    detect_raw = raw["detect"]
    interpret_raw = raw["interpret"]
    quality_gate_raw = raw["quality_gate"]
    report_raw = raw["report"]

    if device_raw["target"] not in DEVICE_TARGETS:
        raise ValueError(
            f"Unknown device.target {device_raw['target']!r}; expected one of {DEVICE_TARGETS}."
        )
    if detect_raw["backend"] not in DETECT_BACKENDS:
        raise ValueError(
            f"Unknown detect.backend {detect_raw['backend']!r}; expected one of {DETECT_BACKENDS}."
        )

    return PipelineConfig(
        paths=PathsConfig(
            data_dir=Path(paths_raw["data_dir"]),
            outputs_dir=Path(paths_raw["outputs_dir"]),
            logs_dir=Path(paths_raw["logs_dir"]),
        ),
        device=DeviceConfig(
            target=device_raw["target"],
            precision=device_raw["precision"],
            batch_size=int(device_raw["batch_size"]),
            torch_device=device_raw["torch_device"],
            seed=int(device_raw["seed"]),
            memory_budget_mb=int(device_raw["memory_budget_mb"]),
            device_total_mb=int(device_raw["device_total_mb"]),
        ),
        ingest=IngestConfig(input_glob=ingest_raw["input_glob"]),
        preprocess=PreprocessConfig(
            image_size=tuple(preprocess_raw["image_size"]),
        ),
        detect=DetectConfig(
            backend=detect_raw["backend"],
            clip_model_name=detect_raw["clip_model_name"],
            clip_cache_dir=Path(detect_raw["clip_cache_dir"]),
            clip_prompt_style=detect_raw["clip_prompt_style"],
            clip_prompts_path=Path(detect_raw["clip_prompts_path"]),
            yolo_model_path=detect_raw["yolo_model_path"],
            yolo_defect_classes=tuple(detect_raw.get("yolo_defect_classes") or ()),
            score_threshold=float(detect_raw["score_threshold"]),
        ),
        interpret=InterpretConfig(
            defect_score_threshold=float(interpret_raw["defect_score_threshold"]),
        ),
        quality_gate=QualityGateConfig(
            green_max_defect_rate=float(quality_gate_raw["green_max_defect_rate"]),
            yellow_max_defect_rate=float(quality_gate_raw["yellow_max_defect_rate"]),
        ),
        report=ReportConfig(title=report_raw["title"]),
    )


def apply_device_target(config: PipelineConfig, target: str) -> PipelineConfig:
    """Apply `--device dev|jetson` overrides on top of a loaded config.

    `target="jetson"` forces `fp16` precision and `batch_size=1` -- the
    on-device deployment path always runs fp16 on a single video stream
    (same reasoning as week 01's `src.config.apply_target_overrides`, just
    also covering batch size here since this pipeline's stages, unlike
    week 01's single benchmark script, batch over a whole `data/` folder
    by default). `target="dev"` is a no-op.

    Args:
        config: A loaded `PipelineConfig`.
        target: One of `DEVICE_TARGETS`.

    Returns:
        `config` unchanged for `"dev"`; a copy with `device.target="jetson"`,
        `device.precision="fp16"` and `device.batch_size=1` for `"jetson"`.

    Raises:
        ValueError: If `target` is not one of `DEVICE_TARGETS`.
    """
    if target not in DEVICE_TARGETS:
        raise ValueError(f"Unknown device target {target!r}; expected one of {DEVICE_TARGETS}.")
    if target == "dev":
        return config
    return replace(
        config,
        device=replace(config.device, target="jetson", precision="fp16", batch_size=1),
    )
