"""Basic neural architecture search -- brute-force grid over a small, bounded space (week 28).

Not AutoML/gradient-based NAS (DARTS, ENAS, etc.) -- a real but
deliberately simple search: train every candidate architecture for the
SAME short budget, evaluate real validation accuracy, pick the best one
under a parameter-count ceiling. Simple, exhaustive, and honest about
being simple -- the roadmap for this week explicitly calls for "базовый
NAS в сильно ограниченном пространстве поиска", not a research-grade
search strategy.
"""

from __future__ import annotations

from dataclasses import dataclass

from distillation.dataset import ClassificationDataset
from distillation.models import TinyCNN, count_parameters
from distillation.training import evaluate_classifier, train_classifier

DEFAULT_SEARCH_SPACE: tuple[tuple[int, ...], ...] = (
    (4,),
    (8,),
    (4, 8),
    (8, 8),
    (8, 16),
    (16, 16),
    (4, 8, 16),
    (8, 16, 16),
)
"""Eight candidates spanning 1-3 conv layers, widths 4-16 -- small enough
to train and evaluate all of them in one demo run, varied enough that
depth and width both actually change across the space."""


@dataclass(frozen=True)
class NasCandidateResult:
    """One architecture's measured outcome.

    Attributes:
        widths: The candidate's `TinyCNN(widths)` architecture.
        num_parameters: Real parameter count.
        accuracy: Real validation accuracy after training.
    """

    widths: tuple[int, ...]
    num_parameters: int
    accuracy: float


@dataclass(frozen=True)
class NasSearchResult:
    """Outcome of `run_nas_search`.

    Attributes:
        all_candidates: Every candidate's `NasCandidateResult`, in search-space order.
        best: The highest-accuracy candidate within `max_parameters`
            (ties broken by fewer parameters -- smaller is still better
            at equal accuracy).
    """

    all_candidates: tuple[NasCandidateResult, ...]
    best: NasCandidateResult


def run_nas_search(
    train_dataset: ClassificationDataset,
    val_dataset: ClassificationDataset,
    max_parameters: int,
    num_steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    search_space: tuple[tuple[int, ...], ...] = DEFAULT_SEARCH_SPACE,
) -> NasSearchResult:
    """Train and evaluate every candidate in `search_space`, pick the best under budget.

    Args:
        train_dataset: Training data, same for every candidate (a fair
            comparison needs identical data/steps/seed across candidates).
        val_dataset: Validation data for scoring.
        max_parameters: Candidates above this parameter count are still
            trained and reported, but never selected as `best`.
        num_steps: Training steps PER CANDIDATE (same budget for all --
            the point of this search is "best architecture for a fixed
            training budget", not "best architecture given unlimited training").
        batch_size: Images per step.
        learning_rate: AdamW learning rate.
        seed: RNG seed, reused identically for every candidate.
        search_space: Candidate architectures to try.

    Returns:
        An `NasSearchResult`.

    Raises:
        ValueError: If no candidate fits within `max_parameters`.
    """
    all_candidates = []
    for widths in search_space:
        model = TinyCNN(widths)
        train_classifier(model, train_dataset, num_steps, batch_size, learning_rate, seed)
        accuracy = evaluate_classifier(model, val_dataset).accuracy
        all_candidates.append(
            NasCandidateResult(
                widths=widths, num_parameters=count_parameters(model), accuracy=accuracy
            )
        )

    in_budget = [c for c in all_candidates if c.num_parameters <= max_parameters]
    if not in_budget:
        raise ValueError(
            f"No candidate in search_space fits within max_parameters={max_parameters}."
        )
    best = max(in_budget, key=lambda c: (c.accuracy, -c.num_parameters))

    return NasSearchResult(all_candidates=tuple(all_candidates), best=best)
