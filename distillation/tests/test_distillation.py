"""Unit tests for `distillation.distillation` -- real training steps, no mocking."""

from __future__ import annotations

from distillation.dataset import build_train_val_split
from distillation.distillation import train_student_with_distillation
from distillation.models import TinyCNN
from distillation.training import train_classifier
from src.synthetic.config import load_defects_config

DEFECTS_CONFIG = load_defects_config("config/defects.yaml")


def _tiny_split():
    return build_train_val_split(
        DEFECTS_CONFIG, num_positive=6, num_negative=4, val_fraction=0.3, seed=42, size=32
    )


def test_train_student_with_distillation_loss_is_real_and_finite() -> None:
    train, _ = _tiny_split()
    teacher = TinyCNN((16, 32))
    train_classifier(teacher, train, num_steps=50, batch_size=4, learning_rate=2e-3, seed=0)

    student = TinyCNN((4, 8))
    run = train_student_with_distillation(
        student, teacher, train, num_steps=10, batch_size=4, learning_rate=1e-3, seed=0
    )
    assert len(run.step_losses) == 10
    assert all(loss == loss for loss in run.step_losses)  # not NaN
    assert all(loss > 0.0 for loss in run.step_losses)


def test_train_student_with_distillation_does_not_update_teacher_weights() -> None:
    train, _ = _tiny_split()
    teacher = TinyCNN((16, 32))
    train_classifier(teacher, train, num_steps=50, batch_size=4, learning_rate=2e-3, seed=0)
    teacher_weights_before = teacher.classifier.weight.detach().clone()

    student = TinyCNN((4, 8))
    train_student_with_distillation(
        student, teacher, train, num_steps=20, batch_size=4, learning_rate=1e-3, seed=0
    )

    assert (teacher.classifier.weight.detach() == teacher_weights_before).all()


def test_train_student_with_distillation_returns_the_same_student_instance() -> None:
    train, _ = _tiny_split()
    teacher = TinyCNN((16, 32))
    train_classifier(teacher, train, num_steps=20, batch_size=4, learning_rate=2e-3, seed=0)

    student = TinyCNN((4, 8))
    run = train_student_with_distillation(
        student, teacher, train, num_steps=5, batch_size=4, learning_rate=1e-3, seed=0
    )
    assert run.model is student
