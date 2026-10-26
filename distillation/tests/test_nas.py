"""Unit tests for `distillation.nas` -- real training steps, no mocking."""

from __future__ import annotations

import pytest

from distillation.dataset import build_train_val_split
from distillation.models import TinyCNN, count_parameters
from distillation.nas import run_nas_search
from src.synthetic.config import load_defects_config

DEFECTS_CONFIG = load_defects_config("config/defects.yaml")

TINY_SEARCH_SPACE = ((2,), (4,), (2, 4))


def _tiny_split():
    return build_train_val_split(
        DEFECTS_CONFIG, num_positive=6, num_negative=4, val_fraction=0.3, seed=42, size=32
    )


def test_run_nas_search_evaluates_every_candidate() -> None:
    train, val = _tiny_split()
    result = run_nas_search(
        train, val, max_parameters=10_000, num_steps=10, batch_size=4, learning_rate=2e-3,
        seed=0, search_space=TINY_SEARCH_SPACE,
    )
    assert len(result.all_candidates) == len(TINY_SEARCH_SPACE)
    assert {c.widths for c in result.all_candidates} == set(TINY_SEARCH_SPACE)


def test_run_nas_search_best_is_within_budget() -> None:
    train, val = _tiny_split()
    result = run_nas_search(
        train, val, max_parameters=10_000, num_steps=10, batch_size=4, learning_rate=2e-3,
        seed=0, search_space=TINY_SEARCH_SPACE,
    )
    assert result.best.num_parameters <= 10_000


def test_run_nas_search_raises_when_nothing_fits_the_budget() -> None:
    train, val = _tiny_split()
    with pytest.raises(ValueError, match="max_parameters"):
        run_nas_search(
            train, val, max_parameters=1, num_steps=5, batch_size=4, learning_rate=2e-3,
            seed=0, search_space=TINY_SEARCH_SPACE,
        )


def test_run_nas_search_never_selects_an_over_budget_candidate_as_best() -> None:
    train, val = _tiny_split()
    # Budget fits only the smallest candidate -- even if a bigger one scored higher.
    small_budget = count_parameters(TinyCNN((2,)))
    result = run_nas_search(
        train, val, max_parameters=small_budget, num_steps=10, batch_size=4, learning_rate=2e-3,
        seed=0, search_space=TINY_SEARCH_SPACE,
    )
    assert result.best.widths == (2,)
