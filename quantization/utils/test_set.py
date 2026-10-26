"""Fixed test-set discovery for `06_trt_inference.py`/`07_evaluate_quality.py`.

Deliberately separate from `03_prepare_calibration_data.py`'s own image
discovery: that script randomly *samples* from a source directory (seeded,
for calibration diversity); this one deterministically *sorts and
truncates* a directory, because 06 and 07 each need the exact same image
set on every invocation -- across different engines (FP16 vs INT4) and
different power modes -- for their numbers to be comparable to each other.
A reseeded random sample would also be reproducible in principle, but
sorted-then-truncated is simpler and doesn't depend on sampling logic
staying identical between the two scripts.
"""

from __future__ import annotations

from pathlib import Path

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def discover_test_images(test_set_dir: Path, max_images: int | None = None) -> list[Path]:
    """Sorted, deterministic list of test images -- the same set for every comparison run.

    Args:
        test_set_dir: Directory of test images (week 06's held-out test split).
        max_images: Cap the count (keeps a comparison run bounded); `None`
            uses every discovered image.

    Returns:
        Sorted image paths, capped at `max_images`.

    Raises:
        FileNotFoundError: If `test_set_dir` doesn't exist.
        ValueError: If no images were found.
    """
    if not test_set_dir.is_dir():
        raise FileNotFoundError(
            f"Test-set directory not found: {test_set_dir}. Point --test-set-dir at a real "
            "image directory -- the default (evaluation.test_set_dir in config.yaml) expects "
            "week 06's dataset_pipeline output; run `cd dataset_pipeline && make pipeline` "
            "first if it doesn't exist yet (see dataset_pipeline/README.md)."
        )
    images = sorted(
        path
        for path in test_set_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise ValueError(f"No test images found under '{test_set_dir}'.")
    return images[:max_images] if max_images is not None else images
