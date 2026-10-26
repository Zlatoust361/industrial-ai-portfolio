"""Zero-shot classification via cosine similarity to averaged prompt embeddings.

Pure numpy logic, decoupled from any specific `VLMInference` backend --
`scripts/clip_classifier.py` is what actually calls `CLIPInference.encode_text`/
`encode_image` and feeds the resulting vectors in here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EDGE_BATCH_SIZE = 1
"""Default classifier batch size: one frame at a time, the on-device streaming scenario."""


@dataclass(frozen=True)
class ClassificationResult:
    """Zero-shot classification outcome for a single image.

    Attributes:
        predicted_class: Class label with the highest cosine similarity.
        similarities: Class label -> cosine similarity to that class's
            reference embedding, for every class considered.
        margin: Similarity of `predicted_class` minus the next-best
            similarity. Larger means a more confident/unambiguous call.
    """

    predicted_class: str
    similarities: dict[str, float]
    margin: float


def average_prompt_embedding(vectors: np.ndarray) -> np.ndarray:
    """Average multiple phrase embeddings into one L2-normalized reference vector.

    Standard prompt-ensembling: several phrasings of the same class (e.g.
    `config/prompts.yaml`'s `defect` phrases) are meant to be combined into
    a single, more robust reference embedding rather than compared
    individually.

    Args:
        vectors: Array of shape `(num_phrases, embedding_dim)`.

    Returns:
        The mean of `vectors` along axis 0, re-normalized to unit L2 norm.

    Raises:
        ValueError: If `vectors` is empty.
    """
    if vectors.shape[0] == 0:
        raise ValueError("Cannot average an empty set of prompt embeddings.")
    mean_vector = vectors.mean(axis=0)
    norm = np.linalg.norm(mean_vector)
    return mean_vector / norm if norm > 0 else mean_vector


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors.

    Does not assume either input is already L2-normalized.

    Args:
        a: First vector, shape `(embedding_dim,)`.
        b: Second vector, shape `(embedding_dim,)`.

    Returns:
        The cosine similarity in `[-1, 1]`, or `0.0` if either vector is all-zero.
    """
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def rank_scores(scores: dict[str, float]) -> tuple[str, float]:
    """Pick the highest-scoring class and its margin over the runner-up.

    Shared by any classification mode that reduces to "one score per class,
    highest wins" -- zero-shot cosine similarity here, and a linear-head
    probability-per-class in `scripts/clip_classifier.py`.

    Args:
        scores: Class label -> score (higher means more likely).

    Returns:
        `(predicted_class, margin)`, where `margin` is the winning score
        minus the runner-up's (or just the winning score, if there's only
        one class).

    Raises:
        ValueError: If `scores` is empty.
    """
    if not scores:
        raise ValueError("rank_scores needs at least one class.")
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    predicted_class, top_score = ranked[0]
    margin = top_score - ranked[1][1] if len(ranked) > 1 else top_score
    return predicted_class, margin


def classify_by_cosine_similarity(
    image_vector: np.ndarray, class_vectors: dict[str, np.ndarray]
) -> ClassificationResult:
    """Classify one image embedding by nearest (cosine) reference class embedding.

    Args:
        image_vector: The image's embedding, shape `(embedding_dim,)`.
        class_vectors: Class label -> reference embedding (e.g. from
            `average_prompt_embedding`), same `embedding_dim`.

    Returns:
        A `ClassificationResult` with the winning class, all per-class
        similarities, and the margin over the runner-up.

    Raises:
        ValueError: If `class_vectors` is empty.
    """
    similarities = {
        label: cosine_similarity(image_vector, class_vector)
        for label, class_vector in class_vectors.items()
    }
    predicted_class, margin = rank_scores(similarities)
    return ClassificationResult(
        predicted_class=predicted_class, similarities=similarities, margin=margin
    )


def describe_batch_size_warning(batch_size: int) -> str | None:
    """Build a warning message for a `--batch-size` greater than 1, if any.

    Zero-shot edge inference on the Jetson Orin Nano reads one camera stream
    at a time -- there's no offline gallery to batch through, so a bigger
    batch size just holds more activation memory without a throughput
    reason to justify it.

    Args:
        batch_size: The requested classifier batch size.

    Returns:
        A human-readable warning if `batch_size > 1`, else `None`.

    Raises:
        ValueError: If `batch_size` is not a positive integer.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size}.")
    if batch_size == 1:
        return None
    return (
        f"--batch-size {batch_size} > 1: zero-shot inference on the Jetson Orin Nano "
        "reads a single camera stream, not an offline gallery, so batching buys no "
        "real throughput here and just uses more memory. Consider --batch-size 1 "
        "unless you're deliberately doing offline bulk processing on a dev machine."
    )
