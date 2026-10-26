"""Unit tests for `src.models.zero_shot`."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.zero_shot import (
    average_prompt_embedding,
    classify_by_cosine_similarity,
    cosine_similarity,
    describe_batch_size_warning,
    rank_scores,
)


def test_average_prompt_embedding_is_unit_norm() -> None:
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
    averaged = average_prompt_embedding(vectors)
    assert np.isclose(np.linalg.norm(averaged), 1.0)
    np.testing.assert_allclose(averaged, [1 / np.sqrt(2), 1 / np.sqrt(2)])


def test_average_prompt_embedding_single_vector_is_unchanged_direction() -> None:
    vectors = np.array([[3.0, 4.0]])
    averaged = average_prompt_embedding(vectors)
    np.testing.assert_allclose(averaged, [0.6, 0.8])


def test_average_prompt_embedding_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        average_prompt_embedding(np.zeros((0, 4)))


def test_cosine_similarity_identical_vectors_is_one() -> None:
    a = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(a, a) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_zero() -> None:
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector_is_zero() -> None:
    assert cosine_similarity(np.zeros(3), np.array([1.0, 2.0, 3.0])) == 0.0


def test_classify_by_cosine_similarity_picks_nearest_class() -> None:
    class_vectors = {"ok": np.array([1.0, 0.0]), "defect": np.array([0.0, 1.0])}

    result = classify_by_cosine_similarity(np.array([0.9, 0.1]), class_vectors)

    assert result.predicted_class == "ok"
    assert result.similarities["ok"] > result.similarities["defect"]
    assert result.margin == pytest.approx(result.similarities["ok"] - result.similarities["defect"])


def test_classify_by_cosine_similarity_empty_classes_raises() -> None:
    with pytest.raises(ValueError, match="at least one class"):
        classify_by_cosine_similarity(np.array([1.0, 0.0]), {})


def test_describe_batch_size_warning_none_for_one() -> None:
    assert describe_batch_size_warning(1) is None


def test_describe_batch_size_warning_present_above_one() -> None:
    warning = describe_batch_size_warning(4)
    assert warning is not None
    assert "4" in warning
    assert "Jetson Orin Nano" in warning


def test_describe_batch_size_warning_rejects_non_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        describe_batch_size_warning(0)


def test_rank_scores_picks_highest_and_computes_margin() -> None:
    predicted_class, margin = rank_scores({"ok": 0.2, "defect": 0.7})
    assert predicted_class == "defect"
    assert margin == pytest.approx(0.5)


def test_rank_scores_single_class_margin_is_its_own_score() -> None:
    predicted_class, margin = rank_scores({"defect": 0.9})
    assert predicted_class == "defect"
    assert margin == pytest.approx(0.9)


def test_rank_scores_empty_raises() -> None:
    with pytest.raises(ValueError, match="at least one class"):
        rank_scores({})
