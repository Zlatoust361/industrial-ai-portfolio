"""Unit tests for `src.models.linear_head`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.models.linear_head import (
    DEFAULT_MAX_FILE_SIZE_KB,
    LinearHead,
    check_linear_head_file_size,
)


def _sample_head(dim: int = 512) -> LinearHead:
    rng = np.random.default_rng(0)
    return LinearHead(
        weights=rng.normal(size=dim),
        bias=-0.1,
        positive_class="defect",
        negative_class="ok",
    )


def test_predict_proba_is_sigmoid_of_linear_score() -> None:
    head = LinearHead(
        weights=np.array([1.0, 0.0]), bias=0.0, positive_class="defect", negative_class="ok"
    )
    probabilities = head.predict_proba(np.array([[0.0, 5.0], [10.0, 0.0]]))
    assert probabilities[0] == pytest.approx(0.5)  # weights . [0, 5] + bias = 0
    assert probabilities[1] > 0.999  # weights . [10, 0] + bias = 10 -> sigmoid ~1


def test_predict_uses_threshold() -> None:
    head = LinearHead(
        weights=np.array([1.0]), bias=0.0, positive_class="defect", negative_class="ok"
    )
    predictions = head.predict(np.array([[-1.0], [1.0]]))
    assert predictions == ["ok", "defect"]


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    head = _sample_head(dim=8)
    path = tmp_path / "head.npz"
    head.save(path)

    loaded = LinearHead.load(path)

    np.testing.assert_allclose(loaded.weights, head.weights)
    assert loaded.bias == pytest.approx(head.bias)
    assert loaded.positive_class == "defect"
    assert loaded.negative_class == "ok"


def test_save_produces_a_kilobyte_sized_file_not_megabytes(tmp_path: Path) -> None:
    # A realistic CLIP embedding dim (e.g. 512-768) should still be a few KB.
    head = _sample_head(dim=768)
    path = tmp_path / "head.npz"
    head.save(path)

    size_kb = path.stat().st_size / 1024
    assert size_kb < 50  # comfortably kilobytes, nowhere near megabytes
    assert size_kb == pytest.approx(check_linear_head_file_size(path))


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        LinearHead.load(tmp_path / "does_not_exist.npz")


def test_check_linear_head_file_size_raises_when_oversized(tmp_path: Path) -> None:
    oversized_path = tmp_path / "oversized.npz"
    oversized_path.write_bytes(b"0" * int((DEFAULT_MAX_FILE_SIZE_KB + 1) * 1024))

    with pytest.raises(ValueError, match="exceeding"):
        check_linear_head_file_size(oversized_path)


def test_check_linear_head_file_size_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        check_linear_head_file_size(tmp_path / "does_not_exist.npz")
