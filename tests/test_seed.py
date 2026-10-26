"""Unit tests for `src.models.seed.set_deterministic_seed` (embedding reproducibility)."""

from __future__ import annotations

import numpy as np

from src.models.seed import set_deterministic_seed


def test_same_seed_reproduces_random_draws() -> None:
    set_deterministic_seed(42)
    first_draw = np.random.rand(16)

    set_deterministic_seed(42)
    second_draw = np.random.rand(16)

    np.testing.assert_array_equal(first_draw, second_draw)


def test_different_seed_changes_random_draws() -> None:
    set_deterministic_seed(1)
    first_draw = np.random.rand(16)

    set_deterministic_seed(2)
    second_draw = np.random.rand(16)

    assert not np.array_equal(first_draw, second_draw)
