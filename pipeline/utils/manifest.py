"""Image manifests: `02_ingest.py` builds one from `<data_dir>/raw/`,
`03_preprocess.py` builds a second one from `<data_dir>/processed/`.

Only needs `PIL` (to validate each file actually decodes and to read its
dimensions) -- no torch/model dependency, unit-testable without either
(same "PIL-only, unit-testable" shape as `ros2_ws/scripts/test_set.py`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImageRecord:
    """One image manifest entry.

    Attributes:
        path: Path to the image file, as a string (JSON-serializable).
        width: Image width in pixels.
        height: Image height in pixels.
    """

    path: str
    width: int
    height: int

    def to_dict(self) -> dict[str, object]:
        """This record as a plain JSON-serializable dict."""
        return asdict(self)


def discover_images(raw_dir: Path, pattern: str) -> list[Path]:
    """List image files under `raw_dir` matching `pattern`, sorted for reproducibility.

    Args:
        raw_dir: Directory to search (non-recursive).
        pattern: Glob pattern, e.g. `"*.jpg"`.

    Returns:
        Matching file paths, sorted by name. Empty if `raw_dir` doesn't
        exist or nothing matches.
    """
    if not raw_dir.is_dir():
        return []
    return sorted(p for p in raw_dir.glob(pattern) if p.is_file())


def build_manifest(paths: list[Path]) -> tuple[list[ImageRecord], list[str]]:
    """Validate each path opens as an image and record its dimensions.

    Args:
        paths: Candidate image file paths.

    Returns:
        `(records, errors)` -- one `ImageRecord` per path that opened
        successfully, and one human-readable error string per path that
        didn't (unreadable/corrupt files are skipped, not fatal, matching
        `ros2_ws/scripts/test_set.py`'s "still run on what's usable"
        stance -- the alternative, failing the whole ingest on one bad
        file, would make a single corrupt image block the entire pipeline
        run).
    """
    from PIL import Image, UnidentifiedImageError

    records: list[ImageRecord] = []
    errors: list[str] = []
    for path in paths:
        try:
            with Image.open(path) as image:
                width, height = image.size
        except (OSError, UnidentifiedImageError) as exc:
            errors.append(f"{path}: {exc}")
            continue
        records.append(ImageRecord(path=str(path), width=width, height=height))
    return records, errors
