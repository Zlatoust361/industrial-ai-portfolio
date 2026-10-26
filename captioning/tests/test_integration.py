"""Tests for `captioning.integration.augment_verdicts_with_captions` -- fully injected, no model."""

from __future__ import annotations

from PIL import Image

from captioning.integration import augment_verdicts_with_captions
from captioning.triggers import TRIGGER_ON_DEFECT_CONFIRMED, TRIGGER_ON_EVERY_FRAME

_VERDICTS = [
    {"image": "a.png", "defect_score": 0.9, "verdict": "defect"},
    {"image": "b.png", "defect_score": 0.1, "verdict": "ok"},
    {"image": "c.png", "defect_score": 0.8, "verdict": "defect"},
]
_IMAGES = {name: Image.new("RGB", (2, 2)) for name in ("a.png", "b.png", "c.png")}


def _fake_caption_fn(image: Image.Image, prompt: str) -> str:
    return "scratch along the edge"


def test_on_defect_confirmed_only_captions_defects() -> None:
    augmented = augment_verdicts_with_captions(
        _VERDICTS,
        load_image=lambda path: _IMAGES[path],
        allowed_phrases=("scratch",),
        model_name="fake-model",
        caption_fn=_fake_caption_fn,
        trigger=TRIGGER_ON_DEFECT_CONFIRMED,
    )

    by_image = {entry["image"]: entry for entry in augmented}
    assert by_image["a.png"]["caption"] == "scratch along the edge"
    assert by_image["c.png"]["caption"] == "scratch along the edge"
    assert by_image["b.png"]["caption"] is None
    assert "caption_latency_ms" in by_image["a.png"]
    assert "caption_latency_ms" not in by_image["b.png"]


def test_on_every_frame_captions_everything() -> None:
    augmented = augment_verdicts_with_captions(
        _VERDICTS,
        load_image=lambda path: _IMAGES[path],
        allowed_phrases=("scratch",),
        model_name="fake-model",
        caption_fn=_fake_caption_fn,
        trigger=TRIGGER_ON_EVERY_FRAME,
    )
    assert all(entry["caption"] is not None for entry in augmented)


def test_does_not_mutate_input_list() -> None:
    original = [dict(v) for v in _VERDICTS]
    augment_verdicts_with_captions(
        _VERDICTS,
        load_image=lambda path: _IMAGES[path],
        allowed_phrases=("scratch",),
        model_name="fake-model",
        caption_fn=_fake_caption_fn,
        trigger=TRIGGER_ON_DEFECT_CONFIRMED,
    )
    assert _VERDICTS == original


def test_preserves_original_fields() -> None:
    augmented = augment_verdicts_with_captions(
        _VERDICTS,
        load_image=lambda path: _IMAGES[path],
        allowed_phrases=("scratch",),
        model_name="fake-model",
        caption_fn=_fake_caption_fn,
        trigger=TRIGGER_ON_DEFECT_CONFIRMED,
    )
    first = augmented[0]
    assert first["defect_score"] == 0.9
    assert first["verdict"] == "defect"
