"""Unit tests for `distillation.quantization_int8` -- real `torch.ao.quantization`, no mocking."""

from __future__ import annotations

import torch

from distillation.dataset import build_train_val_split
from distillation.models import TinyCNN, model_size_kb
from distillation.quantization_int8 import quantize_int8
from distillation.training import evaluate_classifier, train_classifier
from src.synthetic.config import load_defects_config

DEFECTS_CONFIG = load_defects_config("config/defects.yaml")


def _tiny_split():
    return build_train_val_split(
        DEFECTS_CONFIG, num_positive=6, num_negative=4, val_fraction=0.3, seed=42, size=32
    )


def test_quantize_int8_produces_a_smaller_model() -> None:
    train, _ = _tiny_split()
    model = TinyCNN((16, 32))
    train_classifier(model, train, num_steps=30, batch_size=4, learning_rate=2e-3, seed=0)

    quantized = quantize_int8(model, train.images[:4])

    assert model_size_kb(quantized) < model_size_kb(model)


def test_quantize_int8_runs_forward_without_raising() -> None:
    """Real, measured regression this guards against: an earlier version
    of `QuantizableTinyCNN` quantized the classifier Linear too, which
    then received a float (not quantized) input after `dequant` and
    raised `NotImplementedError: Could not run 'quantized::linear' ...
    from the 'CPU' backend` -- see `distillation/quantization_int8.py`'s
    own comment on `quantize_int8`."""
    train, val = _tiny_split()
    model = TinyCNN((8, 16))
    train_classifier(model, train, num_steps=30, batch_size=4, learning_rate=2e-3, seed=0)

    quantized = quantize_int8(model, train.images[:4])
    result = evaluate_classifier(quantized, val)

    assert 0.0 <= result.accuracy <= 1.0


def test_quantize_int8_classifier_and_pool_stay_in_float() -> None:
    train, _ = _tiny_split()
    model = TinyCNN((8,))
    train_classifier(model, train, num_steps=10, batch_size=4, learning_rate=2e-3, seed=0)

    quantized = quantize_int8(model, train.images[:4])

    assert isinstance(quantized.classifier, torch.nn.Linear)
