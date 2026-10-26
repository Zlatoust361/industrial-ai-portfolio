"""Real structured channel pruning for `TinyCNN` (week 28).

"Structured" means whole output channels are removed (not individual
weights zeroed out, which would need sparse-matrix support to actually
save memory/compute) -- the result is a genuinely smaller `TinyCNN`
with fewer real parameters, not the same-sized model with some weights
set to zero.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from distillation.models import TinyCNN


def compute_channel_importance(conv: nn.Conv2d) -> torch.Tensor:
    """L1 norm of each output channel's weights -- the standard, simplest pruning criterion.

    Args:
        conv: A `Conv2d` layer.

    Returns:
        `(out_channels,)` tensor, higher means "more important" (larger
        weight magnitude, a crude but standard proxy for how much a
        channel actually contributes).
    """
    return conv.weight.detach().abs().sum(dim=(1, 2, 3))


def select_kept_channel_indices(importance: torch.Tensor, keep_fraction: float) -> torch.Tensor:
    """Pick the highest-importance channels to keep, in their ORIGINAL order.

    Args:
        importance: Output of `compute_channel_importance`.
        keep_fraction: Fraction of channels to keep, in `(0, 1]`.

    Returns:
        Sorted (ascending, i.e. original-order) indices of the kept channels.

    Raises:
        ValueError: If `keep_fraction` is not in `(0, 1]`.
    """
    if not (0.0 < keep_fraction <= 1.0):
        raise ValueError(f"keep_fraction must be in (0, 1], got {keep_fraction}.")
    num_keep = max(1, round(len(importance) * keep_fraction))
    ranked = torch.argsort(importance, descending=True)[:num_keep]
    return torch.sort(ranked).values


def prune_structured(model: TinyCNN, keep_fraction: float) -> TinyCNN:
    """Build a genuinely smaller `TinyCNN`, keeping only the highest-magnitude channels.

    Handles the chain dependency a real structured prune has to: pruning
    layer `i`'s OUTPUT channels also changes layer `i+1`'s INPUT channels
    (and the final classifier's input width, if the last conv is pruned)
    -- every downstream layer's weights are sliced along both axes to
    stay consistent, not just the pruned layer's own.

    Args:
        model: A trained `TinyCNN`.
        keep_fraction: Fraction of each conv layer's output channels to
            keep (independently per layer).

    Returns:
        A new, smaller `TinyCNN` with copied (not zeroed, not randomly
        re-initialized) weights for every kept channel -- untrained
        weights only exist nowhere, since every kept weight came
        straight from `model`.
    """
    conv_layers = [layer for layer in model.features if isinstance(layer, nn.Conv2d)]

    kept_indices_per_layer = [
        select_kept_channel_indices(compute_channel_importance(conv), keep_fraction)
        for conv in conv_layers
    ]
    new_widths = tuple(len(kept) for kept in kept_indices_per_layer)

    pruned_model = TinyCNN(new_widths, input_channels=conv_layers[0].in_channels)
    pruned_conv_layers = [layer for layer in pruned_model.features if isinstance(layer, nn.Conv2d)]

    previous_kept: torch.Tensor | None = None
    layer_groups = zip(conv_layers, pruned_conv_layers, kept_indices_per_layer, strict=True)
    for conv, pruned_conv, kept in layer_groups:
        weight = conv.weight.detach()[kept]
        if previous_kept is not None:
            weight = weight[:, previous_kept]
        pruned_conv.weight.data.copy_(weight)
        pruned_conv.bias.data.copy_(conv.bias.detach()[kept])
        previous_kept = kept

    pruned_model.classifier.weight.data.copy_(model.classifier.weight.detach()[:, previous_kept])
    pruned_model.classifier.bias.data.copy_(model.classifier.bias.detach())
    return pruned_model
