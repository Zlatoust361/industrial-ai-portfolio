"""Narrow industrial vocabulary prompt + injectable caption generation.

Deliberately NOT open dialogue -- a fixed, small set of allowed defect-
description phrase fragments (config-driven, not hardcoded) is used to
build a constrained instruction prompt, matching the "VQA под узкий
промышленный словарь (не открытый диалог)" skill this week names
explicitly (`course/ROADMAP-WEEKS-13-24.md`'s week 15 entry) rather than
handing the model an open "describe this image" prompt and hoping the
output stays on-topic.

`describe_defect` takes its caption-generating call as an injected
`CaptionFn`, the same injectable-call pattern `assistant/vision_bridge.py`
established for its `client` parameter -- the real call site
(`captioning/scripts/run_captioning_demo.py`) binds it to
`captioning.model.generate_caption_text`; every test here injects a fake
one instead, so this module's own logic (prompt construction, timing,
result shape) is fully testable without a real VLM checkpoint.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from PIL import Image


@dataclass(frozen=True)
class DefectCaption:
    """One generated caption for a confirmed defect.

    Attributes:
        text: The generated description.
        model_name: Identifier of the model that produced it.
        latency_ms: Wall-clock time the generation call took.
    """

    text: str
    model_name: str
    latency_ms: float


class CaptionFn(Protocol):
    """Callable shape `describe_defect` expects for its injected caption call."""

    def __call__(self, image: Image.Image, prompt: str) -> str: ...


def build_prompt(label: str, allowed_phrases: tuple[str, ...]) -> str:
    """Build a constrained instruction prompt for one defect label.

    Args:
        label: The confirmed verdict label (e.g. `"defect"`) -- not the
            specific defect TYPE, which this course's detectors don't
            classify beyond ok/defect; the caption is where a free-text
            description of the SPECIFIC defect actually gets produced.
        allowed_phrases: A small set of standard inspection vocabulary
            fragments (e.g. `("scratch", "crack", "chip along an edge")`)
            given to the model as a hint, not a hard grammar constraint --
            see `captioning/README.md`'s Honesty note for why this is a
            soft (prompted) constraint, not an enforced one.

    Returns:
        The instruction prompt text.
    """
    phrase_hint = ", ".join(allowed_phrases)
    return (
        f"Describe the {label} visible in this industrial part photo in one short "
        f"phrase (max 15 words). Use standard inspection vocabulary, for example: "
        f"{phrase_hint}. Do not speculate beyond what is visible in the image."
    )


def describe_defect(
    image: Image.Image,
    label: str,
    allowed_phrases: tuple[str, ...],
    model_name: str,
    caption_fn: CaptionFn,
) -> DefectCaption:
    """Generate one caption for a confirmed defect, timing the call.

    Args:
        image: The defect image.
        label: Passed to `build_prompt`.
        allowed_phrases: Passed to `build_prompt`.
        model_name: Recorded on the returned `DefectCaption`.
        caption_fn: The injected caption-generating call (see module
            docstring) -- `describe_defect` itself never imports `torch`/
            `transformers`.

    Returns:
        A `DefectCaption` with the generated text and measured latency.
    """
    prompt = build_prompt(label, allowed_phrases)
    start = time.monotonic()
    text = caption_fn(image, prompt)
    latency_ms = (time.monotonic() - start) * 1000.0
    return DefectCaption(text=text, model_name=model_name, latency_ms=latency_ms)
