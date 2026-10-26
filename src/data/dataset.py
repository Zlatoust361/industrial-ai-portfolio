"""Minimal image-folder dataset used for evaluation and benchmarking inputs."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from PIL import Image

from src.data.preprocessing import load_image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

_T = TypeVar("_T")


def iter_batches(items: list[_T], batch_size: int) -> Iterator[list[_T]]:
    """Yield successive `batch_size`-sized slices of `items`.

    Args:
        items: Items to batch, e.g. `dataset.samples`.
        batch_size: Items per yielded slice; the final slice may be shorter.

    Yields:
        Successive slices of `items`, each of length `batch_size` except
        possibly the last.
    """
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


@dataclass(frozen=True)
class ImageSample:
    """A single image sample referenced by path (lazy-loaded)."""

    path: Path

    def load(self) -> Image.Image:
        """Load the underlying image as an RGB PIL `Image`."""
        return load_image(self.path)


class ImageFolderDataset:
    """Enumerates image files under a directory tree, sorted for reproducibility."""

    def __init__(self, root: str | Path) -> None:
        """Args:
            root: Directory scanned recursively for files with an image extension.
        """
        self.root = Path(root)
        self.samples: list[ImageSample] = [
            ImageSample(path=candidate)
            for candidate in sorted(self.root.rglob("*"))
            if candidate.suffix.lower() in IMAGE_EXTENSIONS
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> ImageSample:
        return self.samples[index]


@dataclass(frozen=True)
class LabeledImageSample:
    """A single labeled image sample referenced by path (lazy-loaded).

    Attributes:
        path: Path to the image file.
        label: Ground-truth class label (the name of its parent folder
            under the dataset root, e.g. `"ok"` or `"defect"`).
    """

    path: Path
    label: str

    def load(self) -> Image.Image:
        """Load the underlying image as an RGB PIL `Image`."""
        return load_image(self.path)


class LabeledImageFolderDataset:
    """Enumerates a `root/<class_label>/...` validation set, one folder per class.

    Used for evaluation that needs ground truth (e.g. AUC in
    `scripts/optimize_prompts.py`), unlike the unlabeled `ImageFolderDataset`.
    """

    def __init__(self, root: str | Path, class_labels: list[str]) -> None:
        """Args:
            root: Directory containing one subfolder per label in `class_labels`.
            class_labels: Subfolder names to scan, e.g. `["ok", "defect"]`.

        Raises:
            FileNotFoundError: If any `root/<label>` subfolder is missing.
        """
        self.root = Path(root)
        self.class_labels = list(class_labels)
        self.samples: list[LabeledImageSample] = []

        for label in self.class_labels:
            label_dir = self.root / label
            if not label_dir.is_dir():
                raise FileNotFoundError(f"Expected validation subfolder not found: {label_dir}")
            self.samples.extend(
                LabeledImageSample(path=candidate, label=label)
                for candidate in sorted(label_dir.rglob("*"))
                if candidate.suffix.lower() in IMAGE_EXTENSIONS
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> LabeledImageSample:
        return self.samples[index]
