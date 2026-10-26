"""Unit tests for utils/manifest.py.

Run with (from pipeline/):
    PYTHONPATH=. pytest tests/test_manifest.py
"""

from pathlib import Path

from PIL import Image
from utils.manifest import build_manifest, discover_images


def test_discover_images_matches_glob_and_sorts(tmp_path: Path) -> None:
    for name in ["b.jpg", "a.jpg", "c.png"]:
        (tmp_path / name).write_bytes(b"not a real image, just needs to exist for globbing")

    found = discover_images(tmp_path, "*.jpg")

    assert [p.name for p in found] == ["a.jpg", "b.jpg"]


def test_discover_images_missing_dir_returns_empty() -> None:
    assert discover_images(Path("/nonexistent/raw"), "*.jpg") == []


def test_build_manifest_reads_dimensions(tmp_path: Path) -> None:
    path = tmp_path / "a.jpg"
    Image.new("RGB", (100, 50)).save(path)

    records, errors = build_manifest([path])

    assert errors == []
    assert len(records) == 1
    assert records[0].path == str(path)
    assert records[0].width == 100
    assert records[0].height == 50


def test_build_manifest_skips_unreadable_file_with_error(tmp_path: Path) -> None:
    good_path = tmp_path / "good.jpg"
    Image.new("RGB", (10, 10)).save(good_path)
    bad_path = tmp_path / "bad.jpg"
    bad_path.write_bytes(b"not a real image")

    records, errors = build_manifest([good_path, bad_path])

    assert len(records) == 1
    assert records[0].path == str(good_path)
    assert len(errors) == 1
    assert str(bad_path) in errors[0]


def test_build_manifest_empty_input() -> None:
    assert build_manifest([]) == ([], [])
