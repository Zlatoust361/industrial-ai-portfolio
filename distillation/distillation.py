"""Real knowledge distillation (Hinton et al. 2015-style) for binary classification (week 28).

Standard recipe adapted to a single logit (binary, not a softmax over
many classes): soften both teacher and student logits with a
temperature `T`, match the student's softened probability to the
teacher's via BCE (the binary analogue of KL-divergence-over-softmax
for a single Bernoulli probability), scaled by `T**2` so the gradient
magnitude stays comparable across temperatures (the same scaling the
original paper uses), combined with the usual hard-label loss against
the real ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from distillation.dataset import ClassificationDataset

DEFAULT_TEMPERATURE = 4.0
DEFAULT_ALPHA = 0.5
"""Weight on the hard-label loss; `1 - alpha` weights the soft-label
(distillation) loss -- `0.5` splits them evenly, a common starting point."""


@dataclass(frozen=True)
class DistillationRun:
    """Outcome of `train_student_with_distillation`.

    Attributes:
        model: The trained student.
        step_losses: Real total loss (hard + soft) at every step.
    """

    model: nn.Module
    step_losses: tuple[float, ...]


def train_student_with_distillation(
    student: nn.Module,
    teacher: nn.Module,
    dataset: ClassificationDataset,
    num_steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    temperature: float = DEFAULT_TEMPERATURE,
    alpha: float = DEFAULT_ALPHA,
) -> DistillationRun:
    """Train `student` against both the real labels AND the teacher's softened predictions.

    Args:
        student: An untrained (or already-trained) student model.
        teacher: An ALREADY-TRAINED teacher -- its weights are frozen
            here (`.eval()`, no gradient), never updated by this function.
        dataset: Training data -- same split the teacher itself trained on.
        num_steps: Number of optimizer steps.
        batch_size: Images per step.
        learning_rate: AdamW learning rate.
        seed: RNG seed for batch sampling.
        temperature: Softens both logits before comparison -- higher
            means softer (closer to 0.5), transferring more of the
            teacher's relative uncertainty, not just its hard decision.
        alpha: Hard-label loss weight; `1 - alpha` is the soft-label weight.

    Returns:
        A `DistillationRun` with the trained `student` and every step's
        combined loss.
    """
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(student.parameters(), lr=learning_rate)
    hard_loss_fn = nn.BCEWithLogitsLoss()
    soft_loss_fn = nn.BCELoss()

    teacher.eval()
    student.train()
    step_losses = []
    for _ in range(num_steps):
        batch_indices = rng.integers(0, len(dataset.images), size=batch_size)
        batch_images = dataset.images[batch_indices]
        batch_labels = dataset.labels[batch_indices]

        with torch.no_grad():
            teacher_logits = teacher(batch_images)
        soft_targets = torch.sigmoid(teacher_logits / temperature)

        optimizer.zero_grad()
        student_logits = student(batch_images)
        hard_loss = hard_loss_fn(student_logits, batch_labels)
        soft_predictions = torch.sigmoid(student_logits / temperature)
        soft_loss = soft_loss_fn(soft_predictions, soft_targets) * (temperature**2)

        loss = alpha * hard_loss + (1 - alpha) * soft_loss
        loss.backward()
        optimizer.step()
        step_losses.append(float(loss.item()))

    return DistillationRun(model=student, step_losses=tuple(step_losses))
