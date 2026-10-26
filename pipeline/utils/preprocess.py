"""Aspect-preserving resize ("letterbox") geometry for `03_preprocess.py`.

Pure arithmetic, no PIL/torch -- `03_preprocess.py` is the only place that
actually touches image pixels; this module just computes where they go, so
the placement logic is unit-testable on its own (same "compute the numbers
here, do the I/O in the script" split as `owlvit_postprocess.build_detections`
in week 03).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LetterboxGeometry:
    """Where a resized image goes inside a fixed-size, aspect-preserving canvas.

    Attributes:
        resized_size: `(width, height)` the original image is scaled to,
            preserving its aspect ratio.
        offset: `(x, y)` top-left position to paste `resized_size` at,
            within a `target_size`-sized canvas, so it's centered.
        scale: The scale factor applied to reach `resized_size` (useful for
            mapping detection boxes back to original-image coordinates).
    """

    resized_size: tuple[int, int]
    offset: tuple[int, int]
    scale: float


def compute_letterbox_geometry(
    original_size: tuple[int, int], target_size: tuple[int, int]
) -> LetterboxGeometry:
    """Compute aspect-preserving resize + centered padding to fit `target_size`.

    Scales `original_size` down (or up) by the same factor on both axes so
    it fits entirely inside `target_size`, then centers it -- the standard
    "letterbox" resize used ahead of fixed-input-size vision models, instead
    of a naive per-axis stretch that distorts the image.

    Args:
        original_size: `(width, height)` of the source image.
        target_size: `(width, height)` of the fixed output canvas.

    Returns:
        A `LetterboxGeometry` describing the resize + placement.

    Raises:
        ValueError: If any dimension in `original_size`/`target_size` isn't positive.
    """
    orig_w, orig_h = original_size
    target_w, target_h = target_size
    if orig_w <= 0 or orig_h <= 0 or target_w <= 0 or target_h <= 0:
        raise ValueError(
            f"All dimensions must be positive, got original={original_size}, "
            f"target={target_size}."
        )

    scale = min(target_w / orig_w, target_h / orig_h)
    resized_w = max(1, round(orig_w * scale))
    resized_h = max(1, round(orig_h * scale))
    offset_x = (target_w - resized_w) // 2
    offset_y = (target_h - resized_h) // 2

    return LetterboxGeometry(
        resized_size=(resized_w, resized_h),
        offset=(offset_x, offset_y),
        scale=scale,
    )
