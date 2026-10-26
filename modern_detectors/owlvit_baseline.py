"""Real OWL-ViT zero-shot attempt, for the third point in week 26's comparison.

Reuses the real checkpoint identifier week 03's `ros2_ws/src/vision_bridge/
vision_bridge/detector_factory.py::DEFAULT_MODEL_NAME` already uses
(`google/owlvit-base-patch32`), via `transformers.OwlViTForObjectDetection`/
`OwlViTProcessor` directly (not importing `ros2_ws` itself -- it's a
separate colcon workspace by design, same boundary
`src/evaluation/detection_metrics.py`'s docstring already explains for
week 07's mAP code). Not a mock: a real attempt to load real pretrained
weights.

This sandbox has no cached Hub weights and no outbound access to
`huggingface.co` (same as every other week since week 15's own Honesty
note, and week 25's `classical_cv/clip_baseline.py` for the exact same
reason) -- `run_owlvit_or_blocked` is *expected* to report `blocked`
here, not a classical-CV/RT-DETR/YOLO win by default.
"""

from __future__ import annotations

from dataclasses import dataclass

from modern_detectors.dataset import ResizedImage

OWLVIT_MODEL_NAME = "google/owlvit-base-patch32"
"""Same real checkpoint `ros2_ws`'s week 03 vision_bridge uses."""

DEFECT_QUERIES = ("a scratch", "a pit", "a crack", "a surface defect")
"""Open-vocabulary text queries -- OWL-ViT's whole point (unlike YOLO/
RT-DETR's fixed single `"defect"` class here) is that it needs no
defect-specific training at all, only a text prompt."""

OWLVIT_STATUS_OK = "measured"
OWLVIT_STATUS_BLOCKED = "blocked"


@dataclass(frozen=True)
class OwlVitSummary:
    """OWL-ViT side of the comparison -- either real measured numbers, or an honest `blocked`.

    Attributes:
        status: `"measured"` or `"blocked"`.
        average_precision: AP at the same IoU threshold RT-DETR/YOLO are
            scored at. `None` when `status="blocked"`.
        mean_latency_ms: Mean per-image inference wall-clock time.
            `None` when `status="blocked"`.
        blocked_reason: Human-readable reason, only set when `status="blocked"`.
    """

    status: str
    average_precision: float | None = None
    mean_latency_ms: float | None = None
    blocked_reason: str | None = None


def build_owlvit_model():
    """Build a real OWL-ViT model+processor, loading weights from the Hub.

    Returns:
        `(model, processor)`, both loaded and ready for inference.

    Raises:
        Exception: Whatever `from_pretrained` raises when the checkpoint
            can't be resolved -- in this sandbox, a network/HF-resolution
            failure (`httpx2.ProxyError`/403), not a bug in this module.
            See the module docstring.
    """
    from transformers import OwlViTForObjectDetection, OwlViTProcessor

    processor = OwlViTProcessor.from_pretrained(OWLVIT_MODEL_NAME)
    model = OwlViTForObjectDetection.from_pretrained(OWLVIT_MODEL_NAME).eval()
    return model, processor


def run_owlvit_zero_shot(
    images: list[ResizedImage], model, processor, score_threshold: float = 0.1
) -> dict[str, list]:
    """Run real OWL-ViT zero-shot detection with `DEFECT_QUERIES` on every image.

    Args:
        images: Images to run inference on (`TRAIN_IMAGE_SIZE` each, see
            `modern_detectors.dataset`).
        model: A loaded `OwlViTForObjectDetection` (see `build_owlvit_model`).
        processor: Its matching `OwlViTProcessor`.
        score_threshold: Forwarded to `processor.post_process_object_detection`.

    Returns:
        `image_id -> list[Prediction]` -- same shape as
        `modern_detectors.rt_detr_scratch.predict_rt_detr`'s return value.
    """
    import time

    import torch
    from PIL import Image

    from src.evaluation.detection_metrics import Prediction

    queries = [list(DEFECT_QUERIES)] * len(images)
    pil_images = [Image.fromarray(img.rgb_pixels) for img in images]

    inputs = processor(text=queries, images=pil_images, return_tensors="pt")
    start = time.perf_counter()
    with torch.no_grad():
        outputs = model(**inputs)
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / len(images)

    target_sizes = torch.tensor([img.rgb_pixels.shape[:2] for img in images])
    decoded = processor.post_process_object_detection(
        outputs=outputs, threshold=score_threshold, target_sizes=target_sizes
    )

    predictions: dict[str, list] = {}
    for image, result in zip(images, decoded, strict=True):
        boxes = []
        for score, box in zip(result["scores"], result["boxes"], strict=True):
            x0, y0, x1, y1 = (float(v) for v in box)
            boxes.append(
                Prediction(
                    image_id=image.image_id,
                    class_name="defect",
                    score=float(score),
                    box=(x0, y0, x1, y1),
                )
            )
        predictions[image.image_id] = boxes
    return predictions, elapsed_ms


def run_owlvit_or_blocked(images: list[ResizedImage]) -> OwlVitSummary:
    """Attempt the real OWL-ViT baseline; report a network failure as `blocked`, not a crash.

    Same shape as `classical_cv/clip_baseline.py::run_clip_baseline_or_blocked`
    and `capstone2/report.py::run_step`.

    Args:
        images: Images with ground truth, for AP scoring on success.

    Returns:
        `status="measured"` with real AP/latency on success, or
        `status="blocked"` with the real exception's message on failure
        (expected in this sandbox -- see the module docstring).
    """
    from src.evaluation.detection_metrics import GroundTruth, average_precision

    try:
        model, processor = build_owlvit_model()
        predictions_by_image, latency_ms = run_owlvit_zero_shot(images, model, processor)
    except Exception as exc:  # noqa: BLE001 -- turned into an honest `blocked` status, see docstring
        return OwlVitSummary(
            status=OWLVIT_STATUS_BLOCKED, blocked_reason=f"{type(exc).__name__}: {exc}"
        )

    predictions = [p for preds in predictions_by_image.values() for p in preds]
    ground_truths = [
        GroundTruth(image_id=image.image_id, class_name="defect", box=box)
        for image in images
        for box in image.ground_truth_boxes
    ]
    ap = average_precision(predictions, ground_truths, "defect", iou_threshold=0.5)
    return OwlVitSummary(status=OWLVIT_STATUS_OK, average_precision=ap, mean_latency_ms=latency_ms)
