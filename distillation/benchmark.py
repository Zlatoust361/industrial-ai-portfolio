"""Combine every week 28 compression technique's real measured numbers into one comparison.

Every row in the resulting table is measured the same way: real
accuracy on the same held-out validation split, real per-image CPU
latency, real parameter count, real serialized model size in KiB
(`distillation.models.model_size_kb`) -- comparable across techniques
because the measurement methodology, not just the model, is held fixed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

from distillation.dataset import ClassificationDataset
from distillation.models import count_parameters, model_size_kb
from distillation.training import evaluate_classifier


@dataclass(frozen=True)
class CompressionRow:
    """One technique's real measured numbers, ready for the final comparison table.

    Attributes:
        technique: Human-readable label (`"teacher"`, `"student_plain"`,
            `"student_distilled"`, `"pruned_50pct"`, `"nas_best"`, `"int8_quantized"`).
        num_parameters: Real parameter count.
        size_kb: Real serialized `state_dict` size.
        accuracy: Real validation accuracy.
        latency_ms: Real mean per-image inference latency.
    """

    technique: str
    num_parameters: int
    size_kb: float
    accuracy: float
    latency_ms: float


def measure_row(
    technique: str, model: nn.Module, val_dataset: ClassificationDataset
) -> CompressionRow:
    """Measure one model's real accuracy/latency/size and wrap it as a `CompressionRow`.

    Args:
        technique: Label for the resulting row.
        model: A trained model (any `forward(x) -> (batch, 1)` logit module).
        val_dataset: Held-out validation data.

    Returns:
        A `CompressionRow`.
    """
    result = evaluate_classifier(model, val_dataset)
    return CompressionRow(
        technique=technique,
        num_parameters=count_parameters(model),
        size_kb=model_size_kb(model),
        accuracy=result.accuracy,
        latency_ms=result.mean_latency_ms,
    )
