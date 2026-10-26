"""Unit tests for `distillation.models`."""

from __future__ import annotations

import pytest
import torch

from distillation.models import TinyCNN, count_parameters, model_size_kb


def test_tiny_cnn_forward_shape() -> None:
    model = TinyCNN((4, 8))
    x = torch.rand(3, 1, 16, 16)
    out = model(x)
    assert out.shape == (3, 1)


def test_tiny_cnn_rejects_empty_widths() -> None:
    with pytest.raises(ValueError, match="widths is empty"):
        TinyCNN(())


def test_tiny_cnn_single_layer() -> None:
    model = TinyCNN((4,))
    out = model(torch.rand(1, 1, 8, 8))
    assert out.shape == (1, 1)


def test_count_parameters_matches_manual_count() -> None:
    model = TinyCNN((2,))
    # conv: 2 out * 1 in * 3*3 + 2 bias = 20; classifier: 2*1 + 1 = 3
    assert count_parameters(model) == 20 + 3


def test_model_size_kb_is_positive_and_grows_with_width() -> None:
    small = TinyCNN((4,))
    large = TinyCNN((64,))
    assert model_size_kb(small) > 0.0
    assert model_size_kb(large) > model_size_kb(small)


def test_tiny_cnn_uses_global_max_pool_not_average() -> None:
    """Real, documented architecture choice (see `TinyCNN`'s own
    docstring) -- a regression here would silently reintroduce the
    "every image pools to the same feature vector" bug this week's
    README documents finding."""
    model = TinyCNN((4,))
    assert isinstance(model.pool, torch.nn.AdaptiveMaxPool2d)
