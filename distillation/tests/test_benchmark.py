"""Unit tests for `distillation.benchmark`."""

from __future__ import annotations

from distillation.benchmark import measure_row
from distillation.dataset import build_train_val_split
from distillation.models import TinyCNN, count_parameters, model_size_kb
from distillation.training import train_classifier
from src.synthetic.config import load_defects_config

DEFECTS_CONFIG = load_defects_config("config/defects.yaml")


def test_measure_row_matches_direct_measurements() -> None:
    train, val = build_train_val_split(
        DEFECTS_CONFIG, num_positive=6, num_negative=4, val_fraction=0.3, seed=42, size=32
    )
    model = TinyCNN((4, 8))
    train_classifier(model, train, num_steps=10, batch_size=4, learning_rate=1e-3, seed=0)

    row = measure_row("test_model", model, val)

    assert row.technique == "test_model"
    assert row.num_parameters == count_parameters(model)
    assert row.size_kb == model_size_kb(model)
    assert 0.0 <= row.accuracy <= 1.0
    assert row.latency_ms >= 0.0
