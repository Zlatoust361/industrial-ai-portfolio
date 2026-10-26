"""PromptManager: loads named zero-shot classification prompt styles from YAML.

Reads `config/prompts.yaml` (`basic`/`industrial`/`detailed` phrase sets by
default) so prompt wording lives in config, not hardcoded in
`scripts/clip_classifier.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PromptSet:
    """One named prompt style's phrases, grouped by class label.

    Attributes:
        style: Style name (e.g. `"basic"`, `"industrial"`, `"detailed"`).
        phrases_by_class: Class label (e.g. `"ok"`, `"defect"`) -> list of
            phrases describing that class. Multiple phrases per class are
            meant to be averaged into one reference embedding (see
            `src.models.zero_shot.average_prompt_embedding`), not used
            individually.
    """

    style: str
    phrases_by_class: dict[str, list[str]]

    @property
    def class_labels(self) -> list[str]:
        """Class labels defined for this style, e.g. `["ok", "defect"]`."""
        return list(self.phrases_by_class)


class PromptManager:
    """Loads and serves named prompt styles from a `prompts.yaml` file."""

    def __init__(self, styles: dict[str, PromptSet]) -> None:
        """Args:
            styles: Style name -> `PromptSet`, as produced by `from_yaml`.
        """
        self._styles = styles

    @property
    def available_styles(self) -> list[str]:
        """Style names available from this manager, sorted."""
        return sorted(self._styles)

    def get(self, style: str) -> PromptSet:
        """Look up a prompt style by name.

        Args:
            style: Style name, e.g. `"basic"`, `"industrial"`, `"detailed"`.

        Returns:
            The requested `PromptSet`.

        Raises:
            KeyError: If `style` isn't one of `available_styles`.
        """
        if style not in self._styles:
            raise KeyError(
                f"Unknown prompt style {style!r}; expected one of {self.available_styles}."
            )
        return self._styles[style]

    @classmethod
    def from_yaml(cls, path: str | Path = "config/prompts.yaml") -> PromptManager:
        """Load a `PromptManager` from a `styles:` YAML file.

        Args:
            path: Path to a YAML file with a top-level `styles:` mapping of
                style name -> class label -> list of phrases.

        Returns:
            A `PromptManager` serving every style found in the file.

        Raises:
            FileNotFoundError: If `path` does not exist.
            KeyError: If the file has no top-level `styles:` mapping.
            ValueError: If a style defines no phrases for one of its classes.
        """
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Prompts file not found: {config_path}")

        raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        styles_raw = raw["styles"]

        styles: dict[str, PromptSet] = {}
        for style_name, phrases_by_class in styles_raw.items():
            for class_label, phrases in phrases_by_class.items():
                if not phrases:
                    raise ValueError(
                        f"Style {style_name!r} has no phrases for class {class_label!r}."
                    )
            styles[style_name] = PromptSet(
                style=style_name,
                phrases_by_class={
                    class_label: list(phrases) for class_label, phrases in phrases_by_class.items()
                },
            )
        return cls(styles)
