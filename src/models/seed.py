"""Deterministic-seed helper shared by every `VLMLoader` implementation."""

from __future__ import annotations

import random

import numpy as np


def set_deterministic_seed(seed: int) -> None:
    """Seed Python, NumPy and (if installed) PyTorch RNGs for reproducibility.

    Call this before constructing/initializing a model so that embeddings
    produced by a freshly loaded model are reproducible across runs.

    Args:
        seed: Non-negative integer seed applied to every RNG this process touches.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
