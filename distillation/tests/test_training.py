"""Unit tests for `distillation.training` -- real training steps, no mocking.

Scaled down (few steps, tiny dataset) so the suite stays fast -- `make
demo`'s real, illustrative-scale numbers are documented in
`distillation/README.md`, not re-run here (would take minutes).
"""

from __future__ import annotations

from distillation.dataset import build_train_val_split
from distillation.models import TinyCNN
from distillation.training import evaluate_classifier, train_classifier
from src.synthetic.config import load_defects_config

DEFECTS_CONFIG = load_defects_config("config/defects.yaml")


def _tiny_split():
    return build_train_val_split(
        DEFECTS_CONFIG, num_positive=6, num_negative=4, val_fraction=0.3, seed=42, size=32
    )


def test_train_classifier_loss_is_real_and_finite() -> None:
    train, _ = _tiny_split()
    model = TinyCNN((4, 8))
    run = train_classifier(model, train, num_steps=10, batch_size=4, learning_rate=1e-3, seed=0)
    assert len(run.step_losses) == 10
    assert all(loss == loss for loss in run.step_losses)  # not NaN
    assert all(loss > 0.0 for loss in run.step_losses)


def test_evaluate_classifier_accuracy_in_valid_range() -> None:
    train, val = _tiny_split()
    model = TinyCNN((4, 8))
    train_classifier(model, train, num_steps=10, batch_size=4, learning_rate=1e-3, seed=0)
    result = evaluate_classifier(model, val)
    assert 0.0 <= result.accuracy <= 1.0
    assert result.mean_latency_ms >= 0.0


def test_a_wide_model_can_reach_perfect_accuracy_given_enough_steps() -> None:
    """Real sanity check that the task itself is learnable by this
    architecture family -- not a claim about any particular step budget
    (see `distillation/README.md`'s real numbers for the demo-scale run).
    Uses a bigger tiny split (12+8, not 6+4) than the other tests in this
    file -- empirically, 7 train images is too few for a stable 500-700
    step convergence check (measured flakiness: 300/600/900 steps gave
    0.67/1.0/0.67 accuracy on the smaller split)."""
    train, val = build_train_val_split(
        DEFECTS_CONFIG, num_positive=12, num_negative=8, val_fraction=0.25, seed=42, size=32
    )
    model = TinyCNN((16, 32))
    train_classifier(model, train, num_steps=500, batch_size=4, learning_rate=2e-3, seed=0)
    result = evaluate_classifier(model, val)
    assert result.accuracy == 1.0
