"""Tests for `captioning.evaluation.clip_caption_similarity` -- fake encoder, no real CLIP."""

from __future__ import annotations

import numpy as np
from PIL import Image

from captioning.evaluation import clip_caption_similarity


class _FakeEmbeddingResult:
    def __init__(self, vectors: np.ndarray) -> None:
        self.vectors = vectors


class _FakeInference:
    """Duck-typed stand-in for `src.models.clip.CLIPInference` -- fixed, known vectors."""

    def __init__(self, image_vector: np.ndarray, text_vector: np.ndarray) -> None:
        self._image_vector = image_vector
        self._text_vector = text_vector

    def encode_image(self, images: list[Image.Image]) -> _FakeEmbeddingResult:
        return _FakeEmbeddingResult(np.stack([self._image_vector] * len(images)))

    def encode_text(self, texts: list[str]) -> _FakeEmbeddingResult:
        return _FakeEmbeddingResult(np.stack([self._text_vector] * len(texts)))


def test_identical_normalized_vectors_give_similarity_one() -> None:
    vector = np.array([1.0, 0.0, 0.0])
    inference = _FakeInference(image_vector=vector, text_vector=vector)

    similarity = clip_caption_similarity(inference, Image.new("RGB", (2, 2)), "a caption")
    assert similarity == 1.0


def test_orthogonal_vectors_give_similarity_zero() -> None:
    inference = _FakeInference(
        image_vector=np.array([1.0, 0.0]), text_vector=np.array([0.0, 1.0])
    )

    similarity = clip_caption_similarity(inference, Image.new("RGB", (2, 2)), "a caption")
    assert similarity == 0.0


def test_opposite_vectors_give_similarity_minus_one() -> None:
    inference = _FakeInference(
        image_vector=np.array([1.0, 0.0]), text_vector=np.array([-1.0, 0.0])
    )

    similarity = clip_caption_similarity(inference, Image.new("RGB", (2, 2)), "a caption")
    assert similarity == -1.0
