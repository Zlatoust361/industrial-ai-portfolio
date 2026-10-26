"""When to call captioning -- on a confirmed defect, not on every frame.

Named explicitly in `course/ROADMAP-WEEKS-13-24.md`'s week 15 entry: a
second (generative) model running next to the detector probably can't
share an 8 GB Jetson Orin Nano with it running concurrently every frame
-- call it only when a defect is actually confirmed by the pipeline, not
per frame. `on_every_frame` is kept as an explicit, documented alternative
(not deleted), so the cost of the "wrong" choice is visible in code, not
just prose -- same judgment call `tracking/frame_sampling.py` makes about
stride vs. budget.
"""

from __future__ import annotations

TRIGGER_ON_DEFECT_CONFIRMED = "on_defect_confirmed"
TRIGGER_ON_EVERY_FRAME = "on_every_frame"
VALID_TRIGGERS = (TRIGGER_ON_DEFECT_CONFIRMED, TRIGGER_ON_EVERY_FRAME)


def should_caption(verdict: str, trigger: str) -> bool:
    """Whether this verdict should get a caption call, under `trigger`.

    Args:
        verdict: `"ok"` or `"defect"` (as produced by
            `pipeline.utils.interpret.interpret_detections`, week 04).
        trigger: One of `VALID_TRIGGERS`.

    Returns:
        `True` for every verdict under `on_every_frame`; `True` only for
        `"defect"` under `on_defect_confirmed`.

    Raises:
        ValueError: If `trigger` isn't one of `VALID_TRIGGERS`.
    """
    if trigger not in VALID_TRIGGERS:
        raise ValueError(f"Unknown trigger {trigger!r}; expected one of {VALID_TRIGGERS}.")
    if trigger == TRIGGER_ON_EVERY_FRAME:
        return True
    return verdict == "defect"
