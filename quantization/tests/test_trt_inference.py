"""Unit tests for `utils/trt_inference.py`'s non-TensorRT-dependent pieces.

`cosine_similarity` is pure NumPy -- a real dependency of this sandbox --
so it runs for real here. `TrtImageEncoder`/`benchmark_trt_encoder` need
`tensorrt`/`pycuda` and are only exercised up to that import boundary, via
`test_script_06_trt_inference.py`/`test_script_07_evaluate_quality.py`.
"""

from __future__ import annotations

import numpy as np
import pytest
from utils.trt_inference import cosine_similarity


def test_cosine_similarity_identical_vectors_is_one() -> None:
    a = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    similarities = cosine_similarity(a, a.copy())

    np.testing.assert_allclose(similarities, [1.0, 1.0], atol=1e-6)


def test_cosine_similarity_orthogonal_vectors_is_zero() -> None:
    a = np.array([[1.0, 0.0]], dtype=np.float32)
    b = np.array([[0.0, 1.0]], dtype=np.float32)

    similarities = cosine_similarity(a, b)

    np.testing.assert_allclose(similarities, [0.0], atol=1e-6)


def test_cosine_similarity_opposite_vectors_is_minus_one() -> None:
    a = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    b = -a

    similarities = cosine_similarity(a, b)

    np.testing.assert_allclose(similarities, [-1.0], atol=1e-6)


def test_cosine_similarity_scale_invariant() -> None:
    a = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    b = a * 5.0

    similarities = cosine_similarity(a, b)

    np.testing.assert_allclose(similarities, [1.0], atol=1e-6)


def test_cosine_similarity_shape_mismatch_raises() -> None:
    a = np.zeros((2, 3), dtype=np.float32)
    b = np.zeros((3, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="Shape mismatch"):
        cosine_similarity(a, b)
