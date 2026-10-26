"""Parametric defect-mask generation (week 05).

Renders binary inpainting masks (white/255 = defect area, black/0 =
background) at a given image size, with shape/size/position sampled from a
`DefectsConfig` (see `src.synthetic.config`, `config/defects.yaml`). Pure
`numpy` + `PIL.ImageDraw` -- no `torch`/diffusion dependency; the mask is
day 3's `diffusion_pipeline`'s input, generated entirely dev-side (and,
being cheap 2D drawing, on the Jetson too if ever needed -- unlike
inpainting itself, see `README.md`'s "dev-only" note).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from src.synthetic.config import DefectsConfig, ShapeSpec

Bbox = tuple[int, int, int, int]
"""`(x_min, y_min, x_max, y_max)` pixel bounding box -- half-open, i.e.
`mask[y_min:y_max, x_min:x_max]` covers it exactly, same convention as a
numpy slice. May extend past the image edges (not clipped)."""


@dataclass(frozen=True)
class DefectInstance:
    """One placed defect shape -- traceability for tests/manifests/reports.

    Attributes:
        shape_name: Matches the originating `ShapeSpec.name` (e.g. `"scratch"`).
        shape_type: One of `src.synthetic.config.SHAPE_TYPES`.
        bbox: Pixel bounding box actually drawn, `(x_min, y_min, x_max, y_max)`.
    """

    shape_name: str
    shape_type: str
    bbox: Bbox

    def to_dict(self) -> dict[str, object]:
        """This instance as a plain JSON-serializable dict."""
        return {
            "shape_name": self.shape_name,
            "shape_type": self.shape_type,
            "bbox": list(self.bbox),
        }


def choose_shape(shapes: tuple[ShapeSpec, ...], rng: np.random.Generator) -> ShapeSpec:
    """Weighted-random pick among configured shapes.

    Args:
        shapes: Candidate shapes, e.g. `config.shapes`.
        rng: Source of randomness.

    Returns:
        One `ShapeSpec` from `shapes`.

    Raises:
        ValueError: If `shapes` is empty, or every shape's `weight` is `<= 0`.
    """
    if not shapes:
        raise ValueError("No shapes configured -- defects.yaml's `shapes:` list is empty.")
    weights = np.array([shape.weight for shape in shapes], dtype=float)
    if weights.sum() <= 0:
        raise ValueError("Shape weights must sum to a positive number.")
    index = rng.choice(len(shapes), p=weights / weights.sum())
    return shapes[index]


def sample_position(
    image_size: tuple[int, int], extent: int, margin_fraction: float, rng: np.random.Generator
) -> tuple[int, int]:
    """Sample a center point keeping a shape's full extent within the image margins.

    Args:
        image_size: `(width, height)` of the target image.
        extent: Worst-case pixel size (diameter/length) the shape could occupy,
            centered on the returned point (see `_shape_extent`).
        margin_fraction: Fraction of width/height kept clear of the shape's
            extent, on each edge.
        rng: Source of randomness.

    Returns:
        `(x, y)` pixel coordinates for the shape's center.

    Raises:
        ValueError: If `image_size` is too small for `extent` at the given `margin_fraction`.
    """
    width, height = image_size
    margin_x = int(width * margin_fraction)
    margin_y = int(height * margin_fraction)
    half = extent // 2

    low_x, high_x = margin_x + half, width - margin_x - half
    low_y, high_y = margin_y + half, height - margin_y - half
    if low_x > high_x or low_y > high_y:
        raise ValueError(
            f"Image size {image_size} is too small for shape extent {extent}px at "
            f"margin_fraction={margin_fraction} -- no valid center position exists."
        )
    return int(rng.integers(low_x, high_x + 1)), int(rng.integers(low_y, high_y + 1))


def _shape_extent(shape: ShapeSpec) -> int:
    """Worst-case pixel extent `shape` could occupy, for margin-safe position sampling.

    Uses each range's upper bound rather than the eventual randomly-sampled
    size, so `sample_position` stays correct regardless of what size gets
    drawn afterward (a smaller draw just leaves extra margin, never less).

    `sample_position` treats the returned value as a diameter centered on
    the sampled point (`half = extent // 2` budgeted symmetrically in
    every direction) -- correct for `line` and `ellipse`, both of which
    `_draw_line`/`_draw_ellipse` actually draw centered on that point. Not
    correct for `polyline`: `_draw_polyline`'s random walk *starts* at the
    center and can wander up to its full `total_length` away in one
    direction (bounded turns, so by the triangle inequality its straight-
    line reach from the start is at most `total_length`, never more) --
    budgeting only `length_range[1] // 2` in that direction, as the `line`
    formula would, systematically produces boxes that spill outside the
    image (caught by week 06's dataset validator,
    `dataset_pipeline/scripts/08_validate_dataset.py`).
    """
    if shape.type == "ellipse":
        assert shape.size_range is not None
        return shape.size_range[1]
    if shape.type == "polyline":
        assert shape.length_range is not None and shape.width_range is not None
        return 2 * shape.length_range[1] + shape.width_range[1]
    # line: drawn symmetrically about the center (dx/dy each extend length/2).
    assert shape.length_range is not None and shape.width_range is not None
    return shape.length_range[1] + shape.width_range[1]


def _draw_line(
    draw: ImageDraw.ImageDraw, center: tuple[int, int], shape: ShapeSpec, rng: np.random.Generator
) -> Bbox:
    """Draw a straight stroke of random length/width/angle centered on `center`."""
    assert shape.length_range is not None and shape.width_range is not None
    length = int(rng.integers(shape.length_range[0], shape.length_range[1] + 1))
    width = int(rng.integers(shape.width_range[0], shape.width_range[1] + 1))
    angle = rng.uniform(0, 2 * math.pi)

    dx = round((length / 2) * math.cos(angle))
    dy = round((length / 2) * math.sin(angle))
    x0, y0 = center[0] - dx, center[1] - dy
    x1, y1 = center[0] + dx, center[1] + dy

    draw.line([(x0, y0), (x1, y1)], fill=255, width=width)
    half_width = width // 2 + 1
    return (
        min(x0, x1) - half_width,
        min(y0, y1) - half_width,
        max(x0, x1) + half_width,
        max(y0, y1) + half_width,
    )


def _draw_ellipse(
    draw: ImageDraw.ImageDraw, center: tuple[int, int], shape: ShapeSpec, rng: np.random.Generator
) -> Bbox:
    """Draw a filled ellipse with independently-sampled axis lengths, centered on `center`."""
    assert shape.size_range is not None
    size_x = int(rng.integers(shape.size_range[0], shape.size_range[1] + 1))
    size_y = int(rng.integers(shape.size_range[0], shape.size_range[1] + 1))

    x0, y0 = center[0] - size_x // 2, center[1] - size_y // 2
    x1, y1 = center[0] + size_x // 2, center[1] + size_y // 2
    draw.ellipse([x0, y0, x1, y1], fill=255)
    # PIL's ellipse() bounding box is inclusive of (x1, y1); this Bbox type
    # is half-open (see its docstring), so the upper bound needs a +1.
    return (x0, y0, x1 + 1, y1 + 1)


def _draw_polyline(
    draw: ImageDraw.ImageDraw, center: tuple[int, int], shape: ShapeSpec, rng: np.random.Generator
) -> Bbox:
    """Draw a multi-segment random-walk stroke starting at `center` (crack-like)."""
    assert (
        shape.length_range is not None
        and shape.width_range is not None
        and shape.segments_range is not None
    )
    total_length = rng.integers(shape.length_range[0], shape.length_range[1] + 1)
    segments = int(rng.integers(shape.segments_range[0], shape.segments_range[1] + 1))
    width = int(rng.integers(shape.width_range[0], shape.width_range[1] + 1))
    segment_length = total_length / segments

    points = [(float(center[0]), float(center[1]))]
    angle = rng.uniform(0, 2 * math.pi)
    x, y = float(center[0]), float(center[1])
    for _ in range(segments):
        angle += rng.uniform(-math.pi / 4, math.pi / 4)  # organic bend, not a sharp zigzag
        x += segment_length * math.cos(angle)
        y += segment_length * math.sin(angle)
        points.append((x, y))

    draw.line(points, fill=255, width=width, joint="curve")
    half_width = width // 2 + 1
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (
        int(min(xs)) - half_width,
        int(min(ys)) - half_width,
        int(math.ceil(max(xs))) + half_width,
        int(math.ceil(max(ys))) + half_width,
    )


_DRAW_FUNCS = {"line": _draw_line, "ellipse": _draw_ellipse, "polyline": _draw_polyline}


def generate_defect_mask(
    image_size: tuple[int, int],
    config: DefectsConfig,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, list[DefectInstance]]:
    """Render one binary defect mask.

    Args:
        image_size: `(width, height)` of the target image.
        config: Parsed `defects.yaml` (shapes/weights/count_range/margin).
        rng: Source of randomness; defaults to `np.random.default_rng(config.seed)`
            if not given -- pass your own (seeded once, reused across calls)
            for a reproducible *batch* of distinct masks (see
            `scripts/generate_defect_masks.py`).

    Returns:
        `(mask, instances)`: `mask` is a `(height, width)` `uint8` array,
        values `0`/`255`. `instances` records what was actually drawn, in
        draw order (for tests/manifests/day 4's real-vs-synthetic report).

    Raises:
        ValueError: If `config.shapes` is empty/all-zero-weight (from
            `choose_shape`), or `image_size` can't fit a sampled shape's
            worst-case extent at `config.margin_fraction` (from
            `sample_position`).
    """
    if rng is None:
        rng = np.random.default_rng(config.seed)

    width, height = image_size
    mask_image = Image.new("L", (width, height), color=0)
    draw = ImageDraw.Draw(mask_image)

    count = int(rng.integers(config.count_range[0], config.count_range[1] + 1))
    instances: list[DefectInstance] = []
    for _ in range(count):
        shape = choose_shape(config.shapes, rng)
        center = sample_position(image_size, _shape_extent(shape), config.margin_fraction, rng)
        bbox = _DRAW_FUNCS[shape.type](draw, center, shape, rng)
        instances.append(DefectInstance(shape_name=shape.name, shape_type=shape.type, bbox=bbox))

    return np.array(mask_image, dtype=np.uint8), instances


def mask_coverage_fraction(mask: np.ndarray) -> float:
    """Fraction of `mask`'s pixels that are defect area (non-zero).

    Args:
        mask: A `(height, width)` array as returned by `generate_defect_mask`.

    Returns:
        `defect_pixel_count / total_pixel_count`, in `[0, 1]`. `0.0` for an
        empty (`size == 0`) mask.
    """
    if mask.size == 0:
        return 0.0
    return float(np.count_nonzero(mask)) / float(mask.size)
