"""Augments week 04's pipeline verdicts with a `caption` field -- without touching pipeline/ itself.

Reads the same `verdicts` shape `pipeline.utils.interpret.interpret_detections`
already produces (a list of `{"image", "defect_score", "verdict"}` dicts,
as written to `outputs/interpretations.json`) and adds a `caption` key --
`None` where this week's trigger (`captioning.triggers`) didn't fire.
Writes to THIS week's own output path (see
`captioning/scripts/run_captioning_demo.py`), never `pipeline/`'s --
this week's deliverable extends week 04's report, it never risks
breaking week 04's own tests/outputs by writing into them.
"""

from __future__ import annotations

from collections.abc import Callable

from PIL import Image

from captioning.prompting import CaptionFn, describe_defect
from captioning.triggers import should_caption


def augment_verdicts_with_captions(
    verdicts: list[dict[str, object]],
    load_image: Callable[[str], Image.Image],
    allowed_phrases: tuple[str, ...],
    model_name: str,
    caption_fn: CaptionFn,
    trigger: str,
) -> list[dict[str, object]]:
    """Add a `caption`/`caption_latency_ms` field to each verdict dict.

    Args:
        verdicts: Pipeline verdict dicts (week 04 shape) -- NOT mutated,
            a new list of new dicts is returned.
        load_image: Loads the actual image bytes for one verdict's
            `"image"` path -- injected so this function never hardcodes
            how/where images live on disk (also makes it trivially
            testable with an in-memory fake).
        allowed_phrases: Passed through to `describe_defect`.
        model_name: Passed through to `describe_defect`.
        caption_fn: The injected caption-generating call (see
            `captioning.prompting`'s module docstring).
        trigger: One of `captioning.triggers.VALID_TRIGGERS`.

    Returns:
        A new list, one dict per input verdict, each with `caption`
        (`str | None`) added, and `caption_latency_ms` present only when
        a caption was actually generated.

    Raises:
        ValueError: If `trigger` isn't a recognized value (propagated
            from `captioning.triggers.should_caption`).
        KeyError: If a verdict dict is missing `"verdict"`/`"image"`.
    """
    augmented: list[dict[str, object]] = []
    for entry in verdicts:
        entry = dict(entry)
        verdict = str(entry["verdict"])
        if should_caption(verdict, trigger):
            image = load_image(str(entry["image"]))
            caption = describe_defect(image, verdict, allowed_phrases, model_name, caption_fn)
            entry["caption"] = caption.text
            entry["caption_latency_ms"] = caption.latency_ms
        else:
            entry["caption"] = None
        augmented.append(entry)
    return augmented
