"""Prompt-phrasing sweep: rank candidate phrases by AUC separability (ok vs defect).

Pure numpy/dataclass logic only -- no `torch`/`transformers` import, so it's
unit-testable standalone. `scripts/optimize_prompts.py` is what actually
encodes images/phrases with CLIP and feeds the resulting vectors in here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from src.data.prompts import PromptManager


@dataclass(frozen=True)
class PromptCandidate:
    """One distinct candidate phrase, and every style/class it was sourced from.

    Attributes:
        phrase: The exact phrase text sent to the text encoder.
        origins: `(style, class_label)` pairs this phrase appears under in
            `config/prompts.yaml`. Usually one, but a phrase reused verbatim
            across styles/classes is evaluated once and keeps all origins.
    """

    phrase: str
    origins: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class PromptEvaluation:
    """AUC-separability result for one `PromptCandidate` against a validation set.

    Attributes:
        candidate: The phrase evaluated.
        auc: ROC AUC treating `defect` as the positive class and cosine
            similarity to `candidate.phrase` as the score. 0.5 = no
            information; 1.0 = perfectly ranks defect above ok; 0.0 =
            perfectly ranks ok above defect.
        separability: `max(auc, 1 - auc)` -- how well the phrase separates
            the two classes, regardless of which direction it points.
        polarity: `"higher_similarity_means_defect"` if `auc >= 0.5`, else
            `"higher_similarity_means_ok"`.
    """

    candidate: PromptCandidate
    auc: float
    separability: float
    polarity: str


@dataclass(frozen=True)
class TopKDiff:
    """Comparison of the top-K phrases by separability between two precisions.

    Attributes:
        fp32_phrases: Top-K phrases (highest separability first) in the fp32 pass.
        fp16_phrases: Top-K phrases in the fp16 (`.half()`-emulated) pass.
        same_set: Whether the two top-K lists contain the same phrases (order-independent).
        same_order: Whether the two top-K lists are identical, including rank order.
        entered: Phrases present in `fp16_phrases` but not `fp32_phrases`.
        dropped: Phrases present in `fp32_phrases` but not `fp16_phrases`.
    """

    fp32_phrases: list[str]
    fp16_phrases: list[str]
    same_set: bool
    same_order: bool
    entered: list[str]
    dropped: list[str]


def _average_rank(values: np.ndarray) -> np.ndarray:
    """1-based ascending ranks of `values`, with tied values getting the average rank."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)

    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2  # average of the 1-based positions i+1..j+1
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def compute_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC via the Mann-Whitney U / rank-sum formula (dependency-free, tie-aware).

    Args:
        scores: Predicted score per sample; higher is assumed more "positive".
        labels: Binary ground truth per sample (`1` = positive, `0` = negative),
            same length as `scores`.

    Returns:
        AUC in `[0, 1]`. Returns `0.5` if there are no positive or no
        negative samples (undefined case, treated as "no information").
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)

    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5

    ranks = _average_rank(scores)
    sum_ranks_pos = float(np.sum(ranks[labels == 1]))
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def labels_to_binary(labels: list[str], positive_label: str = "defect") -> np.ndarray:
    """Convert string class labels to a binary array for `compute_auc`.

    Args:
        labels: Class label per sample, e.g. `["ok", "defect", "ok"]`.
        positive_label: The label treated as the positive class (default `"defect"`).

    Returns:
        `int` array of `0`/`1`, same length as `labels`.
    """
    return np.array([1 if label == positive_label else 0 for label in labels], dtype=int)


def collect_candidate_phrases(prompt_manager: PromptManager) -> list[PromptCandidate]:
    """Flatten every style's phrases into a deduplicated list of candidates.

    The same phrase text appearing under multiple styles/classes is merged
    into one `PromptCandidate` (evaluated once), with every origin recorded.

    Args:
        prompt_manager: Loaded `PromptManager` (see `config/prompts.yaml`).

    Returns:
        One `PromptCandidate` per distinct phrase text, in first-seen order.
    """
    candidates_by_phrase: dict[str, PromptCandidate] = {}
    for style in prompt_manager.available_styles:
        prompt_set = prompt_manager.get(style)
        for class_label, phrases in prompt_set.phrases_by_class.items():
            origin = (style, class_label)
            for phrase in phrases:
                existing = candidates_by_phrase.get(phrase)
                if existing is None:
                    candidates_by_phrase[phrase] = PromptCandidate(phrase=phrase, origins=(origin,))
                elif origin not in existing.origins:
                    candidates_by_phrase[phrase] = replace(
                        existing, origins=(*existing.origins, origin)
                    )
    return list(candidates_by_phrase.values())


def evaluate_prompt_candidates(
    image_vectors: np.ndarray,
    labels: np.ndarray,
    candidates: list[PromptCandidate],
    candidate_vectors: dict[str, np.ndarray],
) -> list[PromptEvaluation]:
    """Score every candidate phrase's separability against a validation set.

    Args:
        image_vectors: L2-normalized validation image embeddings, shape `(n_images, dim)`.
        labels: Binary ground truth per image (see `labels_to_binary`), length `n_images`.
        candidates: Phrases to evaluate.
        candidate_vectors: Phrase text -> L2-normalized text embedding, one
            entry per `candidates` entry's `.phrase`.

    Returns:
        One `PromptEvaluation` per candidate, in the same order as `candidates`.
    """
    evaluations = []
    for candidate in candidates:
        vector = candidate_vectors[candidate.phrase]
        scores = image_vectors @ vector
        auc = compute_auc(scores, labels)
        separability = max(auc, 1.0 - auc)
        polarity = "higher_similarity_means_defect" if auc >= 0.5 else "higher_similarity_means_ok"
        evaluations.append(
            PromptEvaluation(
                candidate=candidate, auc=auc, separability=separability, polarity=polarity
            )
        )
    return evaluations


def top_k_by_separability(evaluations: list[PromptEvaluation], k: int) -> list[PromptEvaluation]:
    """The `k` evaluations with the highest `separability`, descending.

    Args:
        evaluations: Candidate evaluations, any order.
        k: Number of top entries to return (fewer if `len(evaluations) < k`).

    Returns:
        Up to `k` `PromptEvaluation`s, sorted by `separability` descending.
    """
    return sorted(evaluations, key=lambda evaluation: evaluation.separability, reverse=True)[:k]


def diff_top_k(fp32_top: list[PromptEvaluation], fp16_top: list[PromptEvaluation]) -> TopKDiff:
    """Compare two precisions' top-K lists to see whether the winners changed.

    Args:
        fp32_top: Top-K evaluations from the fp32 pass (see `top_k_by_separability`).
        fp16_top: Top-K evaluations from the fp16 (`.half()`-emulated) pass.

    Returns:
        A `TopKDiff` describing what changed, if anything.
    """
    fp32_phrases = [evaluation.candidate.phrase for evaluation in fp32_top]
    fp16_phrases = [evaluation.candidate.phrase for evaluation in fp16_top]
    fp32_set, fp16_set = set(fp32_phrases), set(fp16_phrases)

    return TopKDiff(
        fp32_phrases=fp32_phrases,
        fp16_phrases=fp16_phrases,
        same_set=fp32_set == fp16_set,
        same_order=fp32_phrases == fp16_phrases,
        entered=[phrase for phrase in fp16_phrases if phrase not in fp32_set],
        dropped=[phrase for phrase in fp32_phrases if phrase not in fp16_set],
    )
