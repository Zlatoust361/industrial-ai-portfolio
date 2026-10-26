"""Real CLIP zero-shot baseline, for an honest accuracy/latency comparison (week 25).

Reuses week 01/02's real inference stack as-is: `src.models.clip.CLIPLoader`/
`CLIPInference` loading the same `openai/clip-vit-base-patch32` checkpoint
`captioning/`/`explainability/`/`peft/`/`capstone2/` already use, and
`src.models.zero_shot`'s cosine-similarity classifier. Not a mock, not a
simplified reimplementation -- the exact same code path week 01/02 run in
production, pointed at week 25's own synthetic images.

This sandbox has no cached Hub weights and no outbound access to
`huggingface.co` (see every other week's own Honesty note, e.g.
`captioning/README.md`), so `run_clip_zero_shot_baseline` is *expected* to
raise on `CLIPLoader.load()` here -- `classical_cv/scripts/
run_classical_cv_demo.py` catches that and reports it as `blocked`, not as
a classical-CV win by default. `tests/test_clip_baseline.py` asserts this
failure mode directly via `assert_reaches_network_boundary`, the same
helper `captioning/`/`peft/`/`rag/`/`explainability/` already define.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from classical_cv.defect_images import GeneratedImage
from classical_cv.report import CLIP_STATUS_BLOCKED, CLIP_STATUS_OK, ClipBaselineSummary
from src.config import MemorySection, ModelSection
from src.models.clip import CLIPInference, CLIPLoader
from src.models.zero_shot import average_prompt_embedding, classify_by_cosine_similarity

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
"""Same real checkpoint `captioning/`/`explainability/`/`peft/`/`capstone2/` use."""

OK_PROMPTS = ("a photo of a flawless metal part", "a clean part surface with no defects")
DEFECT_PROMPTS = (
    "a photo of a metal part with a scratch or crack",
    "a part surface with a visible defect",
)
"""Two phrasings per class, averaged via `average_prompt_embedding` -- the
same prompt-ensembling `src.models.zero_shot`'s own docstring describes,
not a single hand-picked phrase."""


@dataclass(frozen=True)
class ClipZeroShotResult:
    """Per-image zero-shot call, for the report's per-image table.

    Attributes:
        image_id: Matches `GeneratedImage.image_id`.
        predicted_label: `"ok"` or `"defect"`.
        is_correct: Whether `predicted_label` matches whether the image
            actually has a ground truth defect.
        latency_ms: Wall-clock time of this image's `encode_image` call.
    """

    image_id: str
    predicted_label: str
    is_correct: bool
    latency_ms: float


def _model_config() -> ModelSection:
    return ModelSection(
        name=CLIP_MODEL_NAME,
        type="clip",
        precision="fp32",
        device="cpu",
        seed=42,
        cache_dir=".cache/models",
    )


def _memory_config() -> MemorySection:
    return MemorySection(memory_budget_mb=6144, device_total_mb=8192)


def build_clip_inference() -> CLIPInference:
    """Build a real `CLIPInference`, loading weights from the Hub.

    Returns:
        A loaded `CLIPInference` ready for `encode_image`/`encode_text`.

    Raises:
        Exception: Whatever `CLIPLoader.load()` raises when the checkpoint
            can't be resolved -- in this sandbox, a network/HF-resolution
            failure (`httpx2.ProxyError`/403), not a bug in this module.
            See the module docstring.
    """
    loader = CLIPLoader(_model_config(), _memory_config())
    inference = CLIPInference(loader)
    inference._ensure_loaded()  # noqa: SLF001 -- force the real load now, not lazily later
    return inference


def _as_rgb(pixels: np.ndarray) -> Image.Image:
    """Grayscale `(height, width)` uint8 array -> 3-channel PIL image, as CLIP expects."""
    return Image.fromarray(pixels, mode="L").convert("RGB")


def run_clip_zero_shot_baseline(
    images: list[GeneratedImage], inference: CLIPInference
) -> list[ClipZeroShotResult]:
    """Classify every image as `"ok"`/`"defect"` via real CLIP zero-shot cosine similarity.

    Args:
        images: Dataset built by `classical_cv.defect_images.build_demo_dataset`.
        inference: An already-loaded `CLIPInference` (see `build_clip_inference`).

    Returns:
        One `ClipZeroShotResult` per input image.
    """
    import time

    ok_vectors = inference.encode_text(list(OK_PROMPTS)).vectors
    defect_vectors = inference.encode_text(list(DEFECT_PROMPTS)).vectors
    class_vectors = {
        "ok": average_prompt_embedding(ok_vectors),
        "defect": average_prompt_embedding(defect_vectors),
    }

    results = []
    for image in images:
        start = time.perf_counter()
        image_vector = inference.encode_image([_as_rgb(image.pixels)]).vectors[0]
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        outcome = classify_by_cosine_similarity(image_vector, class_vectors)
        actually_has_defect = len(image.ground_truth_boxes) > 0
        predicted_has_defect = outcome.predicted_class == "defect"

        results.append(
            ClipZeroShotResult(
                image_id=image.image_id,
                predicted_label=outcome.predicted_class,
                is_correct=predicted_has_defect == actually_has_defect,
                latency_ms=elapsed_ms,
            )
        )
    return results


def run_clip_baseline_or_blocked(images: list[GeneratedImage]) -> ClipBaselineSummary:
    """Attempt the real CLIP baseline; report a network failure as `blocked`, not a crash.

    Same shape as `capstone2/report.py::run_step` -- any exception from the
    real attempt (`build_clip_inference`/`run_clip_zero_shot_baseline`)
    becomes an honest `ClipBaselineSummary(status="blocked", ...)` instead
    of propagating and aborting `classical_cv/scripts/run_classical_cv_demo.py`.

    Args:
        images: Dataset built by `classical_cv.defect_images.build_demo_dataset`.

    Returns:
        `status="measured"` with real accuracy/latency on success, or
        `status="blocked"` with the real exception's message on failure
        (expected in this sandbox -- see the module docstring).
    """
    try:
        inference = build_clip_inference()
        results = run_clip_zero_shot_baseline(images, inference)
    except Exception as exc:  # noqa: BLE001 -- turned into an honest `blocked` status, see docstring
        return ClipBaselineSummary(
            status=CLIP_STATUS_BLOCKED, blocked_reason=f"{type(exc).__name__}: {exc}"
        )

    accuracy = sum(1 for r in results if r.is_correct) / len(results)
    mean_latency_ms = sum(r.latency_ms for r in results) / len(results)
    return ClipBaselineSummary(
        status=CLIP_STATUS_OK, accuracy=accuracy, mean_latency_ms=mean_latency_ms
    )
