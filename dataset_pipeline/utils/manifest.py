"""The `Sample` record and manifest JSON I/O shared by every stage (week 06).

Each stage reads the previous stage's manifest and writes a new, more
complete one (`manifest_generated.json` -> `manifest_stratified.json` ->
`manifest_augmented.json` -> `manifest_split.json`) rather than mutating
one file in place -- same "immutable record, one file per stage" shape as
`pipeline/utils/manifest.py` (week 04), so a stage can always be re-run
from its own input without earlier stages' output changing underneath it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

Bbox = tuple[int, int, int, int]
"""`(x_min, y_min, x_max, y_max)`, half-open, same convention as
`src.domain_randomization.scene.Bbox`/`src.synthetic.mask_generation.Bbox`."""


@dataclass(frozen=True)
class Sample:
    """One dataset sample, enriched stage by stage.

    Attributes:
        sample_id: Stable identifier (also the image filename's stem).
        image_path: Path to the sample's image, relative to `data_dir`.
        width: Image width in pixels.
        height: Image height in pixels.
        label: `"ok"` or `"defect"`.
        object_bbox: The rendered object's ground-truth bounding box.
        defect_boxes: Zero or more defect-shape bounding boxes; empty for
            `label == "ok"`.
        dr_params: The domain-randomization render parameters actually
            used for this sample (`RenderParams.to_dict()`) -- stage 2
            reads `light.brightness`/`camera.rotation_deg`/
            `background_color` from here to assign `stratum`.
        stratum: Set by stage 2 (`02_stratify_dataset.py`); `None` before then.
        source_sample_id: For an augmented sample (stage 3), the base
            sample's `sample_id` it was derived from; `None` for a base sample.
        augmentation: For an augmented sample, the augmentation applied
            (e.g. `"hflip"`, `"brightness_jitter"`); `None` for a base sample.
        split: Set by stage 6 (`06_split_dataset.py`); `None` before then.
    """

    sample_id: str
    image_path: str
    width: int
    height: int
    label: str
    object_bbox: Bbox
    defect_boxes: tuple[Bbox, ...]
    dr_params: dict[str, Any]
    stratum: str | None = None
    source_sample_id: str | None = None
    augmentation: str | None = None
    split: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """This sample as a plain JSON-serializable dict."""
        payload = asdict(self)
        payload["object_bbox"] = list(self.object_bbox)
        payload["defect_boxes"] = [list(box) for box in self.defect_boxes]
        return payload


def sample_from_dict(raw: dict[str, Any]) -> Sample:
    """Parse one manifest entry back into a `Sample`.

    Args:
        raw: One `Sample.to_dict()`-shaped dict.

    Returns:
        The parsed `Sample`.
    """
    return Sample(
        sample_id=raw["sample_id"],
        image_path=raw["image_path"],
        width=int(raw["width"]),
        height=int(raw["height"]),
        label=raw["label"],
        object_bbox=tuple(raw["object_bbox"]),
        defect_boxes=tuple(tuple(box) for box in raw["defect_boxes"]),
        dr_params=raw["dr_params"],
        stratum=raw.get("stratum"),
        source_sample_id=raw.get("source_sample_id"),
        augmentation=raw.get("augmentation"),
        split=raw.get("split"),
    )


def load_manifest(path: str | Path) -> list[Sample]:
    """Load a manifest JSON file into a list of `Sample`s.

    Args:
        path: Path to a `{"samples": [...]}` JSON file.

    Returns:
        The parsed samples, in file order.

    Raises:
        FileNotFoundError: If `path` doesn't exist.
    """
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [sample_from_dict(entry) for entry in raw["samples"]]


def save_manifest(path: str | Path, samples: list[Sample]) -> None:
    """Write `samples` to a manifest JSON file, creating parent dirs.

    Args:
        path: Output path.
        samples: Samples to serialize, in the order to write them.
    """
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"samples": [sample.to_dict() for sample in samples]}, indent=2)
    )


def with_stratum(sample: Sample, stratum: str) -> Sample:
    """Return a copy of `sample` with `stratum` set."""
    return replace(sample, stratum=stratum)


def with_split(sample: Sample, split: str) -> Sample:
    """Return a copy of `sample` with `split` set."""
    return replace(sample, split=split)
