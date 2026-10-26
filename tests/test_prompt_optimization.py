"""Unit tests for `src.evaluation.prompt_optimization`."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.prompts import PromptManager, PromptSet
from src.evaluation.prompt_optimization import (
    PromptCandidate,
    PromptEvaluation,
    collect_candidate_phrases,
    compute_auc,
    diff_top_k,
    evaluate_prompt_candidates,
    labels_to_binary,
    top_k_by_separability,
)


def test_compute_auc_perfect_separation_is_one() -> None:
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert compute_auc(scores, labels) == pytest.approx(1.0)


def test_compute_auc_perfectly_reversed_is_zero() -> None:
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([0, 0, 1, 1])
    assert compute_auc(scores, labels) == pytest.approx(0.0)


def test_compute_auc_no_positive_or_negative_samples_is_half() -> None:
    assert compute_auc(np.array([0.1, 0.2, 0.3]), np.array([1, 1, 1])) == 0.5
    assert compute_auc(np.array([0.1, 0.2, 0.3]), np.array([0, 0, 0])) == 0.5


def test_compute_auc_handles_tied_scores() -> None:
    # One tie between a positive (idx 2) and a negative (idx 1) sample at the
    # same score. Pairwise (Mann-Whitney U) count: 3 clear positive-wins + 1
    # half-credit tie, out of 2*2=4 pairs -> 3.5/4 = 0.875.
    scores = np.array([0.1, 0.5, 0.5, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert compute_auc(scores, labels) == pytest.approx(0.875)


def test_labels_to_binary_maps_positive_label_to_one() -> None:
    result = labels_to_binary(["ok", "defect", "ok", "defect"])
    np.testing.assert_array_equal(result, [0, 1, 0, 1])


def test_collect_candidate_phrases_dedupes_and_merges_origins() -> None:
    manager = PromptManager(
        {
            "basic": PromptSet(
                style="basic", phrases_by_class={"defect": ["a flaw"], "ok": ["fine"]}
            ),
            "industrial": PromptSet(
                style="industrial", phrases_by_class={"defect": ["a flaw"], "ok": ["nominal"]}
            ),
        }
    )

    candidates = collect_candidate_phrases(manager)
    by_phrase = {c.phrase: c for c in candidates}

    assert len(candidates) == 3  # "a flaw" deduped, "fine" and "nominal" distinct
    assert set(by_phrase["a flaw"].origins) == {("basic", "defect"), ("industrial", "defect")}
    assert by_phrase["fine"].origins == (("basic", "ok"),)


def test_evaluate_prompt_candidates_and_top_k() -> None:
    # Two images: image 0 is "ok"-like ([1,0]), image 1 is "defect"-like ([0,1]).
    image_vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
    labels = labels_to_binary(["ok", "defect"])

    good_defect_phrase = PromptCandidate(
        phrase="good defect phrase", origins=(("basic", "defect"),)
    )
    useless_phrase = PromptCandidate(phrase="useless phrase", origins=(("basic", "ok"),))
    candidates = [good_defect_phrase, useless_phrase]
    candidate_vectors = {
        "good defect phrase": np.array([0.0, 1.0]),  # matches the defect image exactly
        "useless phrase": np.array([1.0, 1.0]) / np.sqrt(2),  # equidistant, no separability
    }

    evaluations = evaluate_prompt_candidates(image_vectors, labels, candidates, candidate_vectors)
    by_phrase = {e.candidate.phrase: e for e in evaluations}

    assert by_phrase["good defect phrase"].auc == pytest.approx(1.0)
    assert by_phrase["good defect phrase"].separability == pytest.approx(1.0)
    assert by_phrase["good defect phrase"].polarity == "higher_similarity_means_defect"
    assert by_phrase["useless phrase"].separability == pytest.approx(0.5)

    top1 = top_k_by_separability(evaluations, k=1)
    assert len(top1) == 1
    assert top1[0].candidate.phrase == "good defect phrase"


def test_diff_top_k_detects_same_and_changed() -> None:
    def make_eval(phrase: str, separability: float) -> PromptEvaluation:
        return PromptEvaluation(
            candidate=PromptCandidate(phrase=phrase, origins=(("basic", "defect"),)),
            auc=separability,
            separability=separability,
            polarity="higher_similarity_means_defect",
        )

    fp32_top = [make_eval("a", 0.9), make_eval("b", 0.8)]

    same = diff_top_k(fp32_top, [make_eval("a", 0.85), make_eval("b", 0.7)])
    assert same.same_set is True
    assert same.same_order is True
    assert same.entered == []
    assert same.dropped == []

    changed = diff_top_k(fp32_top, [make_eval("a", 0.9), make_eval("c", 0.6)])
    assert changed.same_set is False
    assert changed.same_order is False
    assert changed.entered == ["c"]
    assert changed.dropped == ["b"]

    reordered = diff_top_k(fp32_top, [make_eval("b", 0.95), make_eval("a", 0.6)])
    assert reordered.same_set is True
    assert reordered.same_order is False
