"""Plain real training/evaluation loop for `TinyCNN` -- the baseline every
compression technique in this week is compared against.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn as nn

from distillation.dataset import ClassificationDataset


@dataclass(frozen=True)
class TrainingRun:
    """Outcome of `train_classifier`.

    Attributes:
        model: The trained model.
        step_losses: Real training loss at every step, in order.
    """

    model: nn.Module
    step_losses: tuple[float, ...]


@dataclass(frozen=True)
class EvaluationResult:
    """Outcome of `evaluate_classifier`.

    Attributes:
        accuracy: Fraction of validation images classified correctly at
            a `0.5` probability threshold.
        mean_latency_ms: Mean per-image forward-pass wall-clock time.
    """

    accuracy: float
    mean_latency_ms: float


def train_classifier(
    model: nn.Module,
    dataset: ClassificationDataset,
    num_steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> TrainingRun:
    """Train `model` for real with binary cross-entropy, sampling random batches with replacement.

    Args:
        model: An untrained (or already-trained) `TinyCNN`.
        dataset: Training data (see `distillation.dataset.build_train_val_split`).
        num_steps: Number of optimizer steps.
        batch_size: Images per step.
        learning_rate: AdamW learning rate.
        seed: RNG seed for batch sampling.

    Returns:
        A `TrainingRun` with the trained `model` and every step's real loss.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    step_losses = []
    for _ in range(num_steps):
        batch_indices = rng.integers(0, len(dataset.images), size=batch_size)
        batch_images = dataset.images[batch_indices]
        batch_labels = dataset.labels[batch_indices]

        optimizer.zero_grad()
        logits = model(batch_images)
        loss = loss_fn(logits, batch_labels)
        loss.backward()
        optimizer.step()
        step_losses.append(float(loss.item()))

    return TrainingRun(model=model, step_losses=tuple(step_losses))


def evaluate_classifier(model: nn.Module, dataset: ClassificationDataset) -> EvaluationResult:
    """Measure real accuracy and per-image inference latency on `dataset`.

    Args:
        model: A trained (or untrained, for a sanity check) model.
        dataset: Evaluation data.

    Returns:
        An `EvaluationResult`.
    """
    model.eval()
    correct = 0
    latencies_ms = []
    with torch.no_grad():
        for i in range(len(dataset.images)):
            image = dataset.images[i : i + 1]
            start = time.perf_counter()
            logit = model(image)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)

            predicted = 1.0 if torch.sigmoid(logit).item() >= 0.5 else 0.0
            if predicted == dataset.labels[i].item():
                correct += 1

    accuracy = correct / len(dataset.images) if len(dataset.images) > 0 else 0.0
    mean_latency_ms = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
    return EvaluationResult(accuracy=accuracy, mean_latency_ms=mean_latency_ms)
