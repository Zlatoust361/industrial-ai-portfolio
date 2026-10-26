"""Typed loader for `captioning/config/config.yaml`.

Same shape as `peft/utils/config.py` -- reuses `src.config`'s
`ModelSection`/`MemorySection` directly (this week's captioning model is
loaded through the same `VLMLoader` contract every other week's model is,
see `captioning/model.py::CaptioningLoader`), composed with sections this
week actually adds: prompting (allowed vocabulary), trigger (when to
caption at all), and output paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from captioning.triggers import VALID_TRIGGERS
from src.config import MemorySection, ModelSection


@dataclass(frozen=True)
class PromptingSection:
    """`prompting:` section -- see `captioning.prompting.build_prompt`.

    Attributes:
        allowed_phrases: Small set of standard inspection vocabulary
            fragments hinted to the model -- a SOFT constraint (part of
            the prompt text), not an enforced decoding grammar; see
            `captioning/README.md`'s Honesty note for why.
        max_new_tokens: Generation length cap passed to
            `captioning.model.generate_caption_text`.
    """

    allowed_phrases: tuple[str, ...]
    max_new_tokens: int


@dataclass(frozen=True)
class TriggerSection:
    """`trigger:` section -- see `captioning.triggers`."""

    mode: str


@dataclass(frozen=True)
class PathsSection:
    """`paths:` section -- where `scripts/run_captioning_demo.py` writes its artifacts."""

    outputs_dir: Path
    augmented_report_path: Path
    evaluation_report_path: Path


@dataclass(frozen=True)
class CaptioningConfig:
    """Root config object composed of every section in `config.yaml`."""

    model: ModelSection
    memory: MemorySection
    prompting: PromptingSection
    trigger: TriggerSection
    paths: PathsSection


def load_config(path: str | Path = "config/config.yaml") -> CaptioningConfig:
    """Load and validate the captioning config from a YAML file.

    Args:
        path: Path to a YAML file matching `config/config.yaml`'s schema.

    Returns:
        A fully parsed, immutable `CaptioningConfig`.

    Raises:
        FileNotFoundError: If `path` does not exist.
        KeyError: If a required section or field is missing from the YAML.
        ValueError: If `trigger.mode` isn't one of `captioning.triggers.VALID_TRIGGERS`.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Captioning config file not found: {config_path}")

    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    model_raw = raw["model"]
    memory_raw = raw["memory"]
    prompting_raw = raw["prompting"]
    trigger_raw = raw["trigger"]
    paths_raw = raw["paths"]

    trigger_mode = str(trigger_raw["mode"])
    if trigger_mode not in VALID_TRIGGERS:
        raise ValueError(f"trigger.mode must be one of {VALID_TRIGGERS}, got {trigger_mode!r}.")

    return CaptioningConfig(
        model=ModelSection(
            name=model_raw["name"],
            type=model_raw["type"],
            precision=model_raw["precision"],
            device=model_raw["device"],
            seed=int(model_raw["seed"]),
            cache_dir=Path(model_raw["cache_dir"]),
        ),
        memory=MemorySection(
            memory_budget_mb=int(memory_raw["memory_budget_mb"]),
            device_total_mb=int(memory_raw["device_total_mb"]),
        ),
        prompting=PromptingSection(
            allowed_phrases=tuple(prompting_raw["allowed_phrases"]),
            max_new_tokens=int(prompting_raw["max_new_tokens"]),
        ),
        trigger=TriggerSection(mode=trigger_mode),
        paths=PathsSection(
            outputs_dir=Path(paths_raw["outputs_dir"]),
            augmented_report_path=Path(paths_raw["augmented_report_path"]),
            evaluation_report_path=Path(paths_raw["evaluation_report_path"]),
        ),
    )
