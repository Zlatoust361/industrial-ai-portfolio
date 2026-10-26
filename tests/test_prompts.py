"""Unit tests for `src.data.prompts.PromptManager`."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data.prompts import PromptManager, PromptSet


def test_from_yaml_loads_all_three_styles() -> None:
    manager = PromptManager.from_yaml("config/prompts.yaml")

    assert manager.available_styles == ["basic", "detailed", "industrial"]
    for style in manager.available_styles:
        prompt_set = manager.get(style)
        assert prompt_set.class_labels == ["ok", "defect"]
        assert len(prompt_set.phrases_by_class["ok"]) > 0
        assert len(prompt_set.phrases_by_class["defect"]) > 0


def test_get_unknown_style_raises_key_error() -> None:
    manager = PromptManager.from_yaml("config/prompts.yaml")
    with pytest.raises(KeyError, match="basic"):
        manager.get("nonexistent")


def test_from_yaml_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        PromptManager.from_yaml("config/does_not_exist.yaml")


def test_from_yaml_rejects_empty_class_phrase_list(tmp_path: Path) -> None:
    prompts_path = tmp_path / "prompts.yaml"
    prompts_path.write_text(
        "styles:\n  basic:\n    ok: []\n    defect: ['a defect']\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="basic"):
        PromptManager.from_yaml(prompts_path)


def test_prompt_set_class_labels_preserve_order() -> None:
    prompt_set = PromptSet(style="basic", phrases_by_class={"ok": ["a"], "defect": ["b"]})
    assert prompt_set.class_labels == ["ok", "defect"]
