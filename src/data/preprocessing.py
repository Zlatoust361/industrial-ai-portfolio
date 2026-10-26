"""Framework-free image loading/preprocessing helpers (PIL only, no torch)."""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image


def load_image(path: str | Path) -> Image.Image:
    """Load an image from disk and convert it to RGB.

    Args:
        path: Path to a local image file.

    Returns:
        A PIL `Image` in RGB mode.

    Raises:
        FileNotFoundError: If `path` does not exist.
    """
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return Image.open(image_path).convert("RGB")


def resize_square(image: Image.Image, size: int) -> Image.Image:
    """Resize an image to a `size x size` square using bilinear interpolation.

    Args:
        image: Source image.
        size: Target width/height in pixels.

    Returns:
        The resized image.
    """
    return image.resize((size, size), Image.BILINEAR)


def image_to_png_bytes(image: Image.Image) -> bytes:
    """Serialize a PIL image to in-memory PNG bytes.

    What `VisionBridge.embed_image_bytes`/`classify_by_prompts` (week 11
    day 3) need as raw upload bytes -- a scenario (week 12 day 3) that
    sourced a frame via `src.data.video.sample_frames` (which returns
    PIL images, not bytes) needs exactly this conversion before it can
    call either.

    Args:
        image: Source image (converted to RGB first, so a caller doesn't
            need to worry about RGBA/grayscale/palette modes PNG would
            otherwise encode differently).

    Returns:
        PNG-encoded bytes.
    """
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def parse_image_size(text: str) -> tuple[int, int]:
    """Parse a `"WIDTHxHEIGHT"` CLI argument.

    Args:
        text: e.g. `"512x512"`.

    Returns:
        `(width, height)`.

    Raises:
        ValueError: If `text` isn't exactly two `x`-separated positive integers.
    """
    parts = text.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"Expected WIDTHxHEIGHT (e.g. '512x512'), got {text!r}.")
    width, height = int(parts[0]), int(parts[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"Width/height must be positive, got {(width, height)}.")
    return width, height
