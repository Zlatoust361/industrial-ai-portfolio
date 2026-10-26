"""Thin supervised linear probe on frozen CLIP embeddings.

Trained by `scripts/train_linear_head.py` (`sklearn.linear_model.
LogisticRegression` on CLIP image embeddings); this module only needs numpy
to load the saved weights and run inference with them, so `--use-linear-head`
in `scripts/clip_classifier.py` doesn't pull in scikit-learn at inference
time -- only training does.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_MAX_FILE_SIZE_KB = 64.0
"""Sanity ceiling for a saved `LinearHead` `.npz`.

CLIP embedding dims run a few hundred to ~1536 floats, so a legitimate
weight vector + bias is a handful of KB. Anything past this ceiling means
something bigger than just weights/bias got saved (e.g. a pickled sklearn
estimator instead of its extracted `coef_`/`intercept_`), defeating the
point of a linear probe: negligible on-device memory next to the CLIP
model itself.
"""


@dataclass(frozen=True)
class LinearHead:
    """A binary logistic-regression head trained on frozen CLIP embeddings.

    Attributes:
        weights: Coefficient vector, shape `(embedding_dim,)`.
        bias: Scalar intercept.
        positive_class: Label predicted when `predict_proba(...) >= threshold`.
        negative_class: Label predicted otherwise.
    """

    weights: np.ndarray
    bias: float
    positive_class: str
    negative_class: str

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        """Probability of `positive_class` for each embedding.

        Args:
            embeddings: Array of shape `(n, embedding_dim)`.

        Returns:
            Sigmoid probabilities, shape `(n,)`.
        """
        logits = embeddings @ self.weights + self.bias
        return 1.0 / (1.0 + np.exp(-logits))

    def predict(self, embeddings: np.ndarray, threshold: float = 0.5) -> list[str]:
        """Predicted class label for each embedding.

        Args:
            embeddings: Array of shape `(n, embedding_dim)`.
            threshold: Probability cutoff for `positive_class` (default `0.5`).

        Returns:
            One label per row of `embeddings`.
        """
        probabilities = self.predict_proba(embeddings)
        return [
            self.positive_class if p >= threshold else self.negative_class
            for p in probabilities
        ]

    def save(self, path: str | Path) -> None:
        """Save weights/bias/class labels to a small `.npz` file.

        Args:
            path: Destination path; parent directories are created.
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_path,
            weights=np.asarray(self.weights, dtype=np.float32),
            bias=np.array([self.bias], dtype=np.float32),
            positive_class=np.asarray(self.positive_class),
            negative_class=np.asarray(self.negative_class),
        )

    @classmethod
    def load(cls, path: str | Path) -> LinearHead:
        """Load a `LinearHead` previously written by `save`.

        Args:
            path: Path to a `.npz` file written by `save`.

        Returns:
            The reconstructed `LinearHead`.

        Raises:
            FileNotFoundError: If `path` does not exist.
        """
        head_path = Path(path)
        if not head_path.exists():
            raise FileNotFoundError(f"Linear head weights not found: {head_path}")

        with np.load(head_path, allow_pickle=False) as data:
            return cls(
                weights=data["weights"],
                bias=float(data["bias"][0]),
                positive_class=str(data["positive_class"].item()),
                negative_class=str(data["negative_class"].item()),
            )


def check_linear_head_file_size(
    path: str | Path, max_kb: float = DEFAULT_MAX_FILE_SIZE_KB
) -> float:
    """Assert a saved `LinearHead` file is kilobyte-sized, not megabyte-sized.

    Args:
        path: Path to a `.npz` file written by `LinearHead.save`.
        max_kb: Maximum allowed file size, in KiB.

    Returns:
        The actual file size, in KiB.

    Raises:
        FileNotFoundError: If `path` does not exist.
        ValueError: If the file is larger than `max_kb`.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Linear head weights not found: {file_path}")

    size_kb = file_path.stat().st_size / 1024
    if size_kb > max_kb:
        raise ValueError(
            f"Linear head file '{file_path}' is {size_kb:.1f} KB, exceeding the "
            f"{max_kb:.0f} KB sanity ceiling for a linear probe. This usually means "
            "something bigger than just weights/bias got saved (e.g. a pickled "
            "sklearn estimator) -- check LinearHead.save()."
        )
    return size_kb
