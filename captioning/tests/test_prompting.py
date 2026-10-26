"""Tests for `captioning.prompting` -- pure logic, injected `CaptionFn`, no real model."""

from __future__ import annotations

from PIL import Image

from captioning.prompting import DefectCaption, build_prompt, describe_defect


def test_build_prompt_includes_label_and_phrases() -> None:
    prompt = build_prompt("defect", ("scratch", "crack"))
    assert "defect" in prompt
    assert "scratch" in prompt
    assert "crack" in prompt


def test_describe_defect_returns_caption_from_injected_fn() -> None:
    image = Image.new("RGB", (4, 4), color=(10, 20, 30))
    seen_prompts: list[str] = []

    def fake_caption_fn(img: Image.Image, prompt: str) -> str:
        seen_prompts.append(prompt)
        assert img is image
        return "scratch along the top edge"

    result = describe_defect(
        image, "defect", ("scratch",), "fake-model", caption_fn=fake_caption_fn
    )

    assert isinstance(result, DefectCaption)
    assert result.text == "scratch along the top edge"
    assert result.model_name == "fake-model"
    assert result.latency_ms >= 0.0
    assert len(seen_prompts) == 1
    assert "defect" in seen_prompts[0]


def test_describe_defect_measures_latency() -> None:
    import time

    def slow_caption_fn(img: Image.Image, prompt: str) -> str:
        time.sleep(0.01)
        return "ok"

    result = describe_defect(
        Image.new("RGB", (2, 2)), "defect", (), "fake-model", caption_fn=slow_caption_fn
    )
    assert result.latency_ms >= 10.0
