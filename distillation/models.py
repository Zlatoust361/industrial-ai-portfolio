"""A parametrized tiny CNN -- the shared architecture family for teacher/student/NAS (week 28).

One `TinyCNN` class, width/depth fully parametrized, used for every
role this week needs: the wide "teacher", the narrow "student", and
every candidate `nas.py`'s search evaluates -- a real, honest NAS search
space has to vary architectures of the SAME family for the comparison
to mean anything, not switch architectures entirely between roles.
"""

from __future__ import annotations

import io

import torch
import torch.nn as nn

DEFAULT_INPUT_CHANNELS = 1
TEACHER_WIDTHS = (32, 64, 128)
STUDENT_WIDTHS = (8, 16)
"""Fewer widths (shallower) AND narrower than the teacher -- both axes of
compression at once, same as a real deployed "small model" choice."""


class TinyCNN(nn.Module):
    """A small conv stack: `Conv-ReLU` per width, global MAX pool, one linear head.

    Global MAX pool, not average -- a real, measured finding while
    building this week (see `distillation/README.md`'s Honesty note):
    with `AdaptiveAvgPool2d`, a defect occupying ~1% of the image's area
    gets averaged away into statistical insignificance against a mostly-
    uniform background, and every image (regardless of content) pools
    to nearly the same feature vector -- the model collapses to
    predicting the training set's class prior and nothing else,
    confirmed empirically (every validation probability came out
    identical). `AdaptiveMaxPool2d` instead keeps the single strongest
    activation anywhere in the image, which is exactly the right
    inductive bias for "does ANY region look anomalous" -- switching
    pooling alone took validation accuracy from "majority-class prior"
    to a real, content-dependent prediction.

    Attributes:
        widths: Output channel count of each conv layer, in order --
            `len(widths)` IS the depth; this one tuple fully determines
            the architecture.
    """

    def __init__(
        self, widths: tuple[int, ...], input_channels: int = DEFAULT_INPUT_CHANNELS
    ) -> None:
        """Args:
            widths: Output channels per conv layer, e.g. `(32, 64, 128)`.
            input_channels: Input image channels (`1` for grayscale).

        Raises:
            ValueError: If `widths` is empty.
        """
        super().__init__()
        if not widths:
            raise ValueError("TinyCNN needs at least one conv layer -- widths is empty.")
        self.widths = widths

        layers: list[nn.Module] = []
        in_channels = input_channels
        for width in widths:
            layers.append(nn.Conv2d(in_channels, width, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_channels = width
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Linear(widths[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits, shape `(batch, 1)` -- apply `sigmoid` for a probability."""
        features = self.features(x)
        pooled = self.pool(features).flatten(1)
        return self.classifier(pooled)


def count_parameters(model: nn.Module) -> int:
    """Total number of trainable parameters in `model`."""
    return sum(p.numel() for p in model.parameters())


def model_size_kb(model: nn.Module) -> float:
    """Real serialized size of `model.state_dict()`, in KiB.

    Measured by actually serializing the state dict (`torch.save` to an
    in-memory buffer), not estimated from a parameter-count formula --
    the same "measure the real bytes" standard
    `sla/cost_model.py::measure_prometheus_scrape_bytes` (week 23) holds
    a Prometheus scrape to.
    """
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return len(buffer.getvalue()) / 1024.0
