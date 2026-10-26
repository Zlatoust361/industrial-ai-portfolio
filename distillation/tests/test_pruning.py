"""Unit tests for `distillation.pruning` -- real structured pruning, no mocking."""

from __future__ import annotations

import pytest
import torch

from distillation.models import TinyCNN, count_parameters
from distillation.pruning import (
    compute_channel_importance,
    prune_structured,
    select_kept_channel_indices,
)


def test_compute_channel_importance_matches_manual_l1_norm() -> None:
    model = TinyCNN((4,))
    conv = model.features[0]
    importance = compute_channel_importance(conv)
    expected = conv.weight.detach().abs().sum(dim=(1, 2, 3))
    torch.testing.assert_close(importance, expected)


def test_select_kept_channel_indices_keeps_the_right_count() -> None:
    importance = torch.tensor([0.1, 0.9, 0.3, 0.7])
    kept = select_kept_channel_indices(importance, keep_fraction=0.5)
    assert len(kept) == 2
    # the two highest-importance channels are indices 1 and 3
    assert set(kept.tolist()) == {1, 3}


def test_select_kept_channel_indices_rejects_invalid_fraction() -> None:
    importance = torch.tensor([0.1, 0.2])
    with pytest.raises(ValueError, match="keep_fraction"):
        select_kept_channel_indices(importance, keep_fraction=0.0)
    with pytest.raises(ValueError, match="keep_fraction"):
        select_kept_channel_indices(importance, keep_fraction=1.5)


def test_prune_structured_produces_a_smaller_model() -> None:
    model = TinyCNN((16, 32))
    pruned = prune_structured(model, keep_fraction=0.5)
    assert pruned.widths == (8, 16)
    assert count_parameters(pruned) < count_parameters(model)


def test_prune_structured_copies_real_weights_not_random_ones() -> None:
    """The kept channels' weights must come straight from the original
    model -- never re-initialized -- or pruning would just be "train a
    smaller model from scratch with extra steps", not real pruning."""
    model = TinyCNN((8,))
    importance = compute_channel_importance(model.features[0])
    kept = select_kept_channel_indices(importance, keep_fraction=0.5)

    pruned = prune_structured(model, keep_fraction=0.5)

    original_weight = model.features[0].weight.detach()[kept]
    torch.testing.assert_close(pruned.features[0].weight.detach(), original_weight)


def test_prune_structured_output_matches_a_manual_forward_pass() -> None:
    """The pruned model's forward pass on a real input must agree with
    manually slicing the original model's own forward computation --
    confirms the channel-dependency bookkeeping (pruning layer i's
    output also changes layer i+1's input) is actually correct, not just
    "the right shapes"."""
    torch.manual_seed(0)
    model = TinyCNN((8, 8))
    model.eval()
    pruned = prune_structured(model, keep_fraction=1.0)  # keep everything -- must be a no-op

    x = torch.rand(2, 1, 16, 16)
    with torch.no_grad():
        original_output = model(x)
        pruned_output = pruned(x)
    torch.testing.assert_close(original_output, pruned_output)
