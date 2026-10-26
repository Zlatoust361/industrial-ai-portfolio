"""Per-scenario image sources, dev (synthetic) vs jetson (real camera) (week 12, day 3).

An `image_source` is a zero-argument callable returning one frame's raw
PNG bytes, or `None` when no frame is currently available -- the shape
both scenarios' `run()` expect (see `color_sorting.py`/
`defect_detection.py`). `build_image_source` picks which kind from a
`src.scenarios.config.CameraSource`, resolved from `config/scenarios.yaml`'s
`dev`/`jetson` entries at load time (`src.config.environment`) -- this is
where "конфиг на окружения" actually changes behavior, not just a label.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from src.data.preprocessing import image_to_png_bytes
from src.scenarios.config import CameraSource

_COLOR_RGB: dict[str, tuple[int, int, int]] = {
    "red": (210, 40, 40),
    "green": (40, 170, 70),
    "blue": (40, 90, 200),
    "yellow": (225, 200, 40),
}
"""Approximate reference RGB per color label -- for the synthetic dev
source only; a real camera frame (jetson) needs none of this."""


def _synthesize_colored_image(color: str, size: tuple[int, int], seed: int) -> Image.Image:
    """Deterministic noisy image tinted toward `color` -- no client data.

    Same "smoothed Gaussian noise, not a blank canvas" reasoning as
    `src.synthetic.diffusion_pipeline.synthesize_placeholder_surface`
    (week 05), extended to RGB so a zero-shot color classifier has
    something with real per-channel variation to work against, not a
    flat swatch.

    Args:
        color: A key of `_COLOR_RGB`.
        size: `(width, height)`.
        seed: Seeds a fresh `np.random.default_rng`.

    Returns:
        An RGB `PIL.Image`.

    Raises:
        KeyError: If `color` isn't in `_COLOR_RGB`.
    """
    base = np.array(_COLOR_RGB[color], dtype=np.float64)
    width, height = size
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=18.0, size=(height, width, 3))
    smoothed = gaussian_filter(noise, sigma=(2.0, 2.0, 0.0))
    pixels = np.clip(base + smoothed, 0, 255).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def _synthetic_color_image_source(
    colors: tuple[str, ...], size: tuple[int, int] = (224, 224)
) -> Callable[[], bytes | None]:
    """Cycles deterministically through `colors`, one synthetic frame per call.

    The scenario itself never sees `colors` or which one generated a
    given frame -- it only gets PNG bytes, same as a real camera would
    hand it. A driver script comparing predictions against ground truth
    for a demo report needs to track the cycle index itself.
    """
    state = {"index": 0}

    def _source() -> bytes | None:
        color = colors[state["index"] % len(colors)]
        image = _synthesize_colored_image(color, size, seed=state["index"])
        state["index"] += 1
        return image_to_png_bytes(image)

    return _source


def _synthetic_texture_image_source(
    size: tuple[int, int] = (224, 224),
) -> Callable[[], bytes | None]:
    """One deterministic grayscale-textured frame per call -- `defect_detection.py`'s dev source.

    Reuses week 05's own placeholder-surface generator directly (not a
    second copy) -- the same "no client data, real local texture, not a
    blank canvas" reasoning already applies here.
    """
    from src.synthetic.diffusion_pipeline import synthesize_placeholder_surface

    state = {"index": 0}

    def _source() -> bytes | None:
        image = synthesize_placeholder_surface(size, seed=state["index"])
        state["index"] += 1
        return image_to_png_bytes(image)

    return _source


def _camera_image_source(source: str, max_frames: int = 1) -> Callable[[], bytes | None]:
    """One real frame per call, from a real camera device path or RTSP URL (jetson).

    Args:
        source: Passed straight to `src.data.video.open_video_capture`.
        max_frames: Frames sampled per call before taking the first one
            (matches `src.data.video.sample_frames`'s own signature).

    Returns:
        A callable returning PNG bytes on success, or `None` if the
        capture can't be opened/read (logged by the caller, not raised --
        a scenario loop should keep running past one bad frame).
    """
    from src.data.video import VideoSourceError, open_video_capture, sample_frames

    def _source() -> bytes | None:
        try:
            capture = open_video_capture(source)
            frames = sample_frames(capture, max_frames=max_frames)
        except VideoSourceError:
            return None
        if not frames:
            return None
        return image_to_png_bytes(frames[0])

    return _source


def build_color_sorting_image_source(
    camera: CameraSource, colors: tuple[str, ...]
) -> Callable[[], bytes | None]:
    """The `color_sorting.py` image source for `camera`'s configured environment."""
    if camera.is_synthetic:
        return _synthetic_color_image_source(colors)
    return _camera_image_source(camera.source)


def build_defect_detection_image_source(camera: CameraSource) -> Callable[[], bytes | None]:
    """The `defect_detection.py` image source for `camera`'s configured environment."""
    if camera.is_synthetic:
        return _synthetic_texture_image_source()
    return _camera_image_source(camera.source)
