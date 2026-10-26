#!/usr/bin/env python3
"""Stage 4/7: run the configured detection backend, write `outputs/detections.json`.

Three pluggable backends (`detect.backend` in the config):

- `clip` (default, real): zero-shot ok/defect classification, reusing week
  02's already-tested `src.models.zero_shot`/`src.data.prompts` code
  directly rather than reimplementing zero-shot classification a second
  time -- this stage just batches it over `data/processed/` instead of
  stdout. `defect_score` is a 2-way softmax over the raw ok/defect cosine
  similarities (see `_softmax_defect_probability`'s docstring for why: raw
  CLIP similarities sit around 0.15-0.35, nowhere near a 0.5 threshold).
- `yolo` (real): `ultralytics.YOLO` object detection. A stock
  COCO-pretrained checkpoint has no "defect" class, so
  `detect.yolo_defect_classes` empty means "every detected box counts as a
  defect" -- fine for smoke-testing the wiring, not a real signal; set it
  to a defect-trained checkpoint's actual class name(s) for real use.
- `stub` (CI/demo only): a deterministic, dependency-light pixel-marker
  check -- no torch/transformers/ultralytics at all. This is what
  `run_all.sh`'s default demo run and CI's quality gate exercise, since
  neither has network access to real model weights; see `StubDetector`.

All three lazily import whatever they need (torch/transformers, ultralytics,
or nothing) only inside `detect_batch()` -- this module always imports
cleanly, regardless of what's installed.

`--device jetson` forces `fp16` + `batch_size=1` (see
`utils.config.apply_device_target`, `course/DECISIONS.md` ADR-002 for the
6144 MB / 8192 MB Orin Nano memory-budget default) -- only the `clip`
backend's precision and the batch size images are grouped into actually
change behavior here; `yolo`/`stub` ignore precision (ultralytics/PIL
handle their own dtypes) but still respect `batch_size`.

Input: `data/processed_manifest.json` (from `03_preprocess.py`).
Output: `outputs/detections.json`.

Usage:
    python3 scripts/04_detect.py [--config config/pipeline_config.yaml] [--device dev|jetson]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Protocol

_PIPELINE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PIPELINE_ROOT))
# The `clip` backend reuses week 02's src.models/src.data code from the repo
# root, one level up from pipeline/ -- see this module's docstring.
sys.path.insert(0, str(_PIPELINE_ROOT.parent))

from utils.config import (  # noqa: E402
    DEVICE_TARGETS,
    PipelineConfig,
    apply_device_target,
    load_config,
)


class _Detector(Protocol):
    def detect_batch(self, image_paths: list[Path]) -> list[dict[str, object]]: ...


def _chunk(items: list[Path], size: int) -> list[list[Path]]:
    """Split `items` into consecutive chunks of at most `size`.

    Args:
        items: The list to split.
        size: Maximum chunk length; must be positive.

    Returns:
        A list of chunks, each of length `size` except possibly the last.
    """
    return [items[i : i + size] for i in range(0, len(items), size)]


def _softmax_defect_probability(similarities: dict[str, float]) -> float:
    """Convert raw CLIP ok/defect cosine similarities into a `[0, 1]` defect probability.

    A plain 2-way softmax over the two raw similarities -- not a
    temperature-calibrated probability, just enough structure to give
    `interpret.defect_score_threshold` (a `[0, 1]` threshold, see
    `pipeline_config.yaml`) something meaningful to compare against,
    instead of raw cosine similarities (typically 0.15-0.35 for CLIP,
    nowhere near a 0.5 threshold).

    Args:
        similarities: `{"ok": <cosine similarity>, "defect": <cosine similarity>}`.

    Returns:
        The softmax probability assigned to `"defect"`.
    """
    ok_sim = similarities.get("ok", 0.0)
    defect_sim = similarities.get("defect", 0.0)
    exp_ok = math.exp(ok_sim)
    exp_defect = math.exp(defect_sim)
    return exp_defect / (exp_ok + exp_defect)


class ClipZeroShotDetector:
    """Zero-shot ok/defect classification, reusing week 02's tested CLIP code."""

    def __init__(
        self,
        model_name: str,
        cache_dir: Path,
        prompt_style: str,
        prompts_path: Path,
        precision: str,
        torch_device: str,
        seed: int,
        memory_budget_mb: int,
        device_total_mb: int,
    ) -> None:
        """Args mirror `pipeline_config.yaml`'s `detect:`/`device:` sections."""
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._prompt_style = prompt_style
        self._prompts_path = prompts_path
        self._precision = precision
        self._torch_device = torch_device
        self._seed = seed
        self._memory_budget_mb = memory_budget_mb
        self._device_total_mb = device_total_mb
        self._inference = None
        self._class_vectors: dict[str, object] | None = None

    def _ensure_ready(self) -> None:
        """Build the CLIP loader and reference class embeddings on first use."""
        if self._inference is not None:
            return
        from src.config import MemorySection, ModelSection
        from src.data.prompts import PromptManager
        from src.models.clip import CLIPInference, CLIPLoader
        from src.models.zero_shot import average_prompt_embedding

        model_section = ModelSection(
            name=self._model_name,
            type="clip",
            precision=self._precision,
            device=self._torch_device,
            seed=self._seed,
            cache_dir=self._cache_dir,
        )
        memory_section = MemorySection(
            memory_budget_mb=self._memory_budget_mb, device_total_mb=self._device_total_mb
        )
        loader = CLIPLoader(model_section, memory_section)
        self._inference = CLIPInference(loader)

        prompt_manager = PromptManager.from_yaml(self._prompts_path)
        prompt_set = prompt_manager.get(self._prompt_style)

        self._class_vectors = {
            class_label: average_prompt_embedding(self._inference.encode_text(phrases).vectors)
            for class_label, phrases in prompt_set.phrases_by_class.items()
        }

    def detect_batch(self, image_paths: list[Path]) -> list[dict[str, object]]:
        """Classify a batch of images as ok/defect via one `encode_image` call.

        Real batching (not a loop calling a single-image method): all
        images in `image_paths` go through `CLIPInference.encode_image` in
        one forward pass -- `device.batch_size` (forced to 1 on `--device
        jetson`, see this module's docstring) controls how large that batch
        is, straight from `04_detect.py::main`'s chunking.
        """
        from PIL import Image

        from src.models.zero_shot import classify_by_cosine_similarity

        self._ensure_ready()
        assert self._inference is not None and self._class_vectors is not None

        images = []
        for path in image_paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        image_result = self._inference.encode_image(images)

        results = []
        for vector in image_result.vectors:
            result = classify_by_cosine_similarity(vector, self._class_vectors)
            results.append(
                {
                    "defect_score": _softmax_defect_probability(result.similarities),
                    "predicted_class": result.predicted_class,
                    "similarities": result.similarities,
                }
            )
        return results


class YoloDetector:
    """`ultralytics.YOLO` object detection, defect_score from the flagged classes' confidence."""

    def __init__(
        self, model_path: str, score_threshold: float, defect_classes: tuple[str, ...]
    ) -> None:
        """Args mirror `pipeline_config.yaml`'s `detect:` section."""
        self._model_path = model_path
        self._score_threshold = score_threshold
        self._defect_classes = set(defect_classes)
        self._model = None

    def _ensure_ready(self) -> None:
        """Load the YOLO checkpoint on first use."""
        if self._model is not None:
            return
        from ultralytics import YOLO

        self._model = YOLO(self._model_path)

    def detect_batch(self, image_paths: list[Path]) -> list[dict[str, object]]:
        """Run YOLO on a batch of images; `ultralytics` accepts a list source natively."""
        self._ensure_ready()
        assert self._model is not None

        results = self._model.predict(source=[str(p) for p in image_paths], verbose=False)

        per_image = []
        for result in results:
            boxes: list[dict[str, object]] = []
            defect_score = 0.0
            for box in result.boxes:
                confidence = float(box.conf[0])
                if confidence < self._score_threshold:
                    continue
                class_name = result.names[int(box.cls[0])]
                boxes.append({"class_name": class_name, "confidence": confidence})
                if not self._defect_classes or class_name in self._defect_classes:
                    defect_score = max(defect_score, confidence)
            per_image.append({"defect_score": defect_score, "boxes": boxes})
        return per_image


class StubDetector:
    """Deterministic, dependency-light stand-in detector for `run_all.sh`'s demo run and CI.

    Looks for a bright-red marker rectangle (RGB ~ `(220, 30, 30)`) painted
    onto an image by `scripts/generate_demo_dataset.py` -- a real, if
    trivial, per-pixel signal rather than a hardcoded constant, so it
    genuinely exercises `05_interpret.py`/`06_quality_gate.py`/
    `07_generate_report.py` against real image content without needing
    torch/transformers/ultralytics or any model weights at all. NOT a
    substitute for `clip`/`yolo` in a real inspection run -- see
    `pipeline/README.md`'s "CI/demo backend" section.
    """

    _MARKER_RGB = (220, 30, 30)
    _MARKER_TOLERANCE = 30
    _MARKER_MIN_FRACTION = 0.01  # >=1% of pixels close to the marker color -> "defect"

    def detect_batch(self, image_paths: list[Path]) -> list[dict[str, object]]:
        """Flag each image as `defect` if enough of its pixels match the marker color."""
        import numpy as np
        from PIL import Image

        target = np.array(self._MARKER_RGB, dtype=np.int16)
        results = []
        for path in image_paths:
            with Image.open(path) as image:
                pixels = np.asarray(image.convert("RGB"), dtype=np.int16)
            close = np.all(np.abs(pixels - target) <= self._MARKER_TOLERANCE, axis=-1)
            marker_fraction = float(close.mean())
            defect_score = 0.9 if marker_fraction >= self._MARKER_MIN_FRACTION else 0.05
            results.append({"defect_score": defect_score, "marker_fraction": marker_fraction})
        return results


def build_detector(config: PipelineConfig) -> _Detector:
    """Construct the configured `detect.backend` detector.

    Args:
        config: The loaded pipeline config.

    Returns:
        A `ClipZeroShotDetector`, `YoloDetector`, or `StubDetector`, per
        `config.detect.backend`.
    """
    if config.detect.backend == "clip":
        return ClipZeroShotDetector(
            model_name=config.detect.clip_model_name,
            cache_dir=config.detect.clip_cache_dir,
            prompt_style=config.detect.clip_prompt_style,
            prompts_path=config.detect.clip_prompts_path,
            precision=config.device.precision,
            torch_device=config.device.torch_device,
            seed=config.device.seed,
            memory_budget_mb=config.device.memory_budget_mb,
            device_total_mb=config.device.device_total_mb,
        )
    if config.detect.backend == "yolo":
        return YoloDetector(
            model_path=config.detect.yolo_model_path,
            score_threshold=config.detect.score_threshold,
            defect_classes=config.detect.yolo_defect_classes,
        )
    return StubDetector()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `04_detect.py`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/pipeline_config.yaml")
    parser.add_argument(
        "--device",
        choices=DEVICE_TARGETS,
        default=None,
        help="Override device.target (jetson forces fp16 + batch_size=1). "
        "Default: use the config's device.target as-is.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the configured detector over every processed image, write `outputs/detections.json`."""
    args = parse_args()
    config = load_config(args.config)
    if args.device is not None:
        config = apply_device_target(config, args.device)

    processed_manifest_path = config.paths.data_dir / "processed_manifest.json"
    if not processed_manifest_path.is_file():
        print(
            f"Error: processed manifest not found at '{processed_manifest_path}' -- "
            "run 03_preprocess.py first.",
            file=sys.stderr,
        )
        sys.exit(1)
    processed_manifest = json.loads(processed_manifest_path.read_text())

    detector = build_detector(config)
    image_paths = [Path(entry["processed_path"]) for entry in processed_manifest["images"]]

    detections = []
    for chunk in _chunk(image_paths, config.device.batch_size):
        for image_path, result in zip(chunk, detector.detect_batch(chunk), strict=True):
            detections.append({"image": str(image_path), **result})

    config.paths.outputs_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.paths.outputs_dir / "detections.json"
    output_path.write_text(
        json.dumps({"backend": config.detect.backend, "detections": detections}, indent=2)
    )

    print(f"Detected {len(detections)} image(s) with backend '{config.detect.backend}'.")
    print(f"Detections written to '{output_path}'.")


if __name__ == "__main__":
    main()
