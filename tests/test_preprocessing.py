"""Unit tests for `src.data.preprocessing`."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from src.data.preprocessing import image_to_png_bytes, load_image, parse_image_size, resize_square


def test_resize_square_returns_requested_dimensions() -> None:
    image = Image.new("RGB", (64, 32))
    resized = resize_square(image, size=224)
    assert resized.size == (224, 224)


def test_load_image_converts_to_rgb(tmp_path: Path) -> None:
    source = Image.new("L", (16, 16))
    image_path = tmp_path / "sample.png"
    source.save(image_path)

    loaded = load_image(image_path)
    assert loaded.mode == "RGB"
    assert loaded.size == (16, 16)


def test_load_image_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "missing.png")


def test_parse_image_size_parses_width_and_height() -> None:
    assert parse_image_size("512x512") == (512, 512)
    assert parse_image_size("300X200") == (300, 200)  # case-insensitive


def test_parse_image_size_rejects_bad_format() -> None:
    with pytest.raises(ValueError, match="WIDTHxHEIGHT"):
        parse_image_size("512")


def test_parse_image_size_rejects_non_positive_dimensions() -> None:
    with pytest.raises(ValueError, match="positive"):
        parse_image_size("0x100")


def test_image_to_png_bytes_produces_a_real_decodable_png() -> None:
    from io import BytesIO

    image = Image.new("RGB", (8, 8), color=(255, 0, 0))
    png_bytes = image_to_png_bytes(image)

    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"  # real PNG magic bytes
    decoded = Image.open(BytesIO(png_bytes))
    assert decoded.size == (8, 8)
    assert decoded.format == "PNG"


def test_image_to_png_bytes_converts_non_rgb_modes() -> None:
    from io import BytesIO

    image = Image.new("L", (4, 4))
    png_bytes = image_to_png_bytes(image)
    assert Image.open(BytesIO(png_bytes)).convert("RGB").mode == "RGB"
