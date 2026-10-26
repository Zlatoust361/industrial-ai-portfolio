"""Typed loader for `config/diffusion.yaml` (week 05, day 3: inpainting + quality gate).

Same "typed YAML section -> frozen dataclass" shape as
`src/synthetic/config.py` (week 05, day 2) -- kept as its own module for
the same reason that one gives for not folding into `src/config.py`:
inpainting-backend/quality-gate settings have nothing to do with defect
mask geometry, even though both live under `config/` and `src/synthetic/`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

BACKEND_NAMES: tuple[str, ...] = ("cv2", "diffusers", "stability_api")
"""Valid `backend:` values -- see `src.synthetic.diffusion_pipeline.build_backend`."""


@dataclass(frozen=True)
class PromptConfig:
    """`prompt:` section -- shared text conditioning for the diffusion backends."""

    positive: str
    negative: str


@dataclass(frozen=True)
class Cv2BackendConfig:
    """`cv2:` section -- OpenCV's classical inpainting parameters."""

    method: str
    radius: int


@dataclass(frozen=True)
class DiffusersBackendConfig:
    """`diffusers:` section -- local Stable Diffusion inpainting via HuggingFace `diffusers`."""

    model_id: str
    device: str
    num_inference_steps: int
    guidance_scale: float
    seed: int


@dataclass(frozen=True)
class StabilityAPIBackendConfig:
    """`stability_api:` section -- Stability AI's REST inpainting API."""

    endpoint: str
    api_key_env: str
    output_format: str
    timeout_s: float


@dataclass(frozen=True)
class QualityGateConfig:
    """`quality_gate:` section -- how the 4 metrics combine into a keep/reject decision.

    Attributes:
        keep_fraction: Fraction of a batch to keep, by composite score,
            highest first. In `[0, 1]`.
        weights: Per-metric weight, keyed by the same names
            `quality_assessment.assess()` returns (`psnr_outside_mask`,
            `ssim_outside_mask`, `blur_inside_mask`,
            `contrast_inside_mask`). Relative, not required to sum to 1 --
            normalized at combine time (see
            `quality_assessment.composite_score`).
        ssim_window: Uniform-window size (px) for the simplified SSIM
            implementation in `quality_assessment.py`.
    """

    keep_fraction: float
    weights: dict[str, float]
    ssim_window: int


@dataclass(frozen=True)
class DiffusionConfig:
    """Root config object composed of every section in `diffusion.yaml`."""

    backend: str
    prompt: PromptConfig
    cv2: Cv2BackendConfig
    diffusers: DiffusersBackendConfig
    stability_api: StabilityAPIBackendConfig
    quality_gate: QualityGateConfig


def load_diffusion_config(path: str | Path = "config/diffusion.yaml") -> DiffusionConfig:
    """Load and validate the inpainting/quality-gate config from a YAML file.

    Args:
        path: Path to a YAML file matching `diffusion.yaml`'s schema.

    Returns:
        A fully parsed, immutable `DiffusionConfig`.

    Raises:
        FileNotFoundError: If `path` does not exist.
        KeyError: If a required field is missing.
        ValueError: If `backend` isn't one of `BACKEND_NAMES`, or
            `quality_gate.keep_fraction` isn't in `[0, 1]`.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Diffusion config file not found: {config_path}")

    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    backend = raw["backend"]
    if backend not in BACKEND_NAMES:
        raise ValueError(f"Unknown backend {backend!r}; expected one of {BACKEND_NAMES}.")

    keep_fraction = float(raw["quality_gate"]["keep_fraction"])
    if not 0.0 <= keep_fraction <= 1.0:
        raise ValueError(f"quality_gate.keep_fraction must be in [0, 1], got {keep_fraction}.")

    return DiffusionConfig(
        backend=backend,
        prompt=PromptConfig(
            positive=str(raw["prompt"]["positive"]).strip(),
            negative=str(raw["prompt"]["negative"]).strip(),
        ),
        cv2=Cv2BackendConfig(
            method=str(raw["cv2"]["method"]),
            radius=int(raw["cv2"]["radius"]),
        ),
        diffusers=DiffusersBackendConfig(
            model_id=str(raw["diffusers"]["model_id"]),
            device=str(raw["diffusers"]["device"]),
            num_inference_steps=int(raw["diffusers"]["num_inference_steps"]),
            guidance_scale=float(raw["diffusers"]["guidance_scale"]),
            seed=int(raw["diffusers"]["seed"]),
        ),
        stability_api=StabilityAPIBackendConfig(
            endpoint=str(raw["stability_api"]["endpoint"]),
            api_key_env=str(raw["stability_api"]["api_key_env"]),
            output_format=str(raw["stability_api"]["output_format"]),
            timeout_s=float(raw["stability_api"]["timeout_s"]),
        ),
        quality_gate=QualityGateConfig(
            keep_fraction=keep_fraction,
            weights={str(k): float(v) for k, v in raw["quality_gate"]["weights"].items()},
            ssim_window=int(raw["quality_gate"]["ssim_window"]),
        ),
    )
