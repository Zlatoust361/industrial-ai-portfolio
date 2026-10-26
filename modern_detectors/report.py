"""Combine RT-DETR-tiny, YOLOv8n-scratch and the OWL-ViT attempt into one honest report (week 26).

Same honesty standard `classical_cv/report.py` (week 25) holds: the OWL-ViT
side is reported as `"blocked"` when the real attempt hits this sandbox's
network boundary, never silently omitted or guessed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from modern_detectors.benchmark import DetectorBenchReport
from modern_detectors.owlvit_baseline import OwlVitSummary


@dataclass(frozen=True)
class ModernDetectorsReport:
    """The week 26 deliverable: RT-DETR-tiny vs YOLOv8n-scratch vs OWL-ViT, side by side.

    Attributes:
        rt_detr: Real measured AP/precision/recall/latency.
        yolo: Real measured AP/precision/recall/latency, same dataset, same scoring.
        owlvit: Real measured numbers, or an honest `blocked`.
        recommendation: One sentence grounded in this run's own numbers.
    """

    rt_detr: DetectorBenchReport
    yolo: DetectorBenchReport
    owlvit: OwlVitSummary
    recommendation: str


def _recommend(
    rt_detr: DetectorBenchReport, yolo: DetectorBenchReport, owlvit: OwlVitSummary
) -> str:
    """One sentence, grounded in this run's own numbers -- not a canned opinion."""
    faster_name = (
        "RT-DETR-tiny" if rt_detr.mean_latency_ms <= yolo.mean_latency_ms else "YOLOv8n-scratch"
    )
    better_ap_name = (
        "RT-DETR-tiny" if rt_detr.average_precision >= yolo.average_precision else "YOLOv8n-scratch"
    )

    base = (
        f"At this toy scale (both trained from scratch, same {rt_detr.num_images} images, "
        f"same IoU={rt_detr.iou_threshold} scoring): "
        f"RT-DETR-tiny AP={rt_detr.average_precision:.2f} @ "
        f"{rt_detr.mean_latency_ms:.1f} ms/image; "
        f"YOLOv8n-scratch AP={yolo.average_precision:.2f} @ {yolo.mean_latency_ms:.1f} ms/image "
        f"on CPU -- {better_ap_name} scored higher AP, {faster_name} ran faster in this run."
    )
    if owlvit.status == "blocked":
        return (
            f"{base} OWL-ViT zero-shot blocked ({owlvit.blocked_reason}) -- open-vocabulary "
            "detection needs no training data at all when it works, which is exactly the "
            "comparison this sandbox can't complete; weigh it once a real network/cached "
            "checkpoint is available, don't assume it from these numbers."
        )
    return (
        f"{base} OWL-ViT zero-shot: AP={owlvit.average_precision:.2f} @ "
        f"{owlvit.mean_latency_ms:.1f} ms/image, with ZERO training images -- "
        "weigh that against RT-DETR/YOLO's real training cost, not just their AP."
    )


def build_report(
    rt_detr: DetectorBenchReport, yolo: DetectorBenchReport, owlvit: OwlVitSummary
) -> ModernDetectorsReport:
    """Assemble the final week 26 comparison report."""
    return ModernDetectorsReport(
        rt_detr=rt_detr, yolo=yolo, owlvit=owlvit, recommendation=_recommend(rt_detr, yolo, owlvit)
    )


def to_dict(report: ModernDetectorsReport) -> dict[str, object]:
    """JSON-serializable view of `report`."""
    return {
        "rt_detr": asdict(report.rt_detr),
        "yolo": asdict(report.yolo),
        "owlvit": asdict(report.owlvit),
        "recommendation": report.recommendation,
    }
