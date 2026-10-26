"""CLI: defect classification over a folder of images via CLIP.

Two classification modes:

- Default (zero-shot): cosine similarity against per-class reference
  embeddings built by averaging a prompt style's phrases (see
  `config/prompts.yaml`, `src.data.prompts.PromptManager`).
- `--use-linear-head PATH`: a trained `LinearHead` (see
  `scripts/train_linear_head.py`) scores each image directly from its CLIP
  embedding instead -- no prompts involved.

Either way, reports the per-image scores and the winning class.

Usage:
    PYTHONPATH=. python scripts/clip_classifier.py --input-dir path/to/images \\
        [--style basic|industrial|detailed] [--batch-size 1] \\
        [--config config/model_config.yaml] [--prompts config/prompts.yaml] \\
        [--use-linear-head outputs/week02/linear_head.npz] \\
        [--output outputs/week02/classification_report.json]

`--batch-size` defaults to 1: zero-shot inference on the Jetson Orin Nano
reads a single camera stream (not an offline gallery), so batching just
spends memory without a real throughput reason. Passing a larger value
still works, but prints a warning explaining why it's usually unnecessary.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from src.config import load_config
from src.data.dataset import ImageFolderDataset, iter_batches
from src.data.prompts import PromptManager
from src.models.clip import CLIPInference, CLIPLoader
from src.models.linear_head import LinearHead
from src.models.zero_shot import (
    EDGE_BATCH_SIZE,
    average_prompt_embedding,
    classify_by_cosine_similarity,
    describe_batch_size_warning,
    rank_scores,
)

ScoreFn = Callable[[np.ndarray], dict[str, float]]


def build_class_reference_vectors(
    inference: CLIPInference, phrases_by_class: dict[str, list[str]]
) -> dict[str, np.ndarray]:
    """Encode and average each class's phrases into one reference embedding per class.

    Args:
        inference: A loaded (or lazily-loading) `CLIPInference`.
        phrases_by_class: Class label -> phrases describing that class
            (typically `PromptSet.phrases_by_class`).

    Returns:
        Class label -> L2-normalized reference embedding.
    """
    class_vectors: dict[str, np.ndarray] = {}
    for class_label, phrases in phrases_by_class.items():
        text_embeddings = inference.encode_text(phrases)
        class_vectors[class_label] = average_prompt_embedding(text_embeddings.vectors)
    return class_vectors


def make_zero_shot_scorer(class_vectors: dict[str, np.ndarray]) -> ScoreFn:
    """Build a scorer that ranks classes by cosine similarity to `class_vectors`."""

    def score(image_vector: np.ndarray) -> dict[str, float]:
        return classify_by_cosine_similarity(image_vector, class_vectors).similarities

    return score


def make_linear_head_scorer(head: LinearHead) -> ScoreFn:
    """Build a scorer that ranks classes by a trained `LinearHead`'s probability."""

    def score(image_vector: np.ndarray) -> dict[str, float]:
        probability = float(head.predict_proba(image_vector[np.newaxis, :])[0])
        return {head.positive_class: probability, head.negative_class: 1.0 - probability}

    return score


def classify_folder(
    inference: CLIPInference,
    dataset: ImageFolderDataset,
    batch_size: int,
    score_fn: ScoreFn,
) -> list[dict[str, Any]]:
    """Run batched classification over every image in `dataset`.

    Args:
        inference: A loaded (or lazily-loading) `CLIPInference`.
        dataset: Images to classify.
        batch_size: Images per `encode_image` call.
        score_fn: Maps one image embedding to `{class_label: score}` --
            see `make_zero_shot_scorer`/`make_linear_head_scorer`.

    Returns:
        One record per image, in dataset order: `{"path", "predicted_class",
        "scores", "margin"}`.
    """
    records: list[dict[str, Any]] = []
    for batch in iter_batches(dataset.samples, batch_size):
        images = [sample.load() for sample in batch]
        embeddings = inference.encode_image(images)
        for sample, image_vector in zip(batch, embeddings.vectors, strict=True):
            scores = score_fn(image_vector)
            predicted_class, margin = rank_scores(scores)
            records.append(
                {
                    "path": str(sample.path),
                    "predicted_class": predicted_class,
                    "scores": scores,
                    "margin": margin,
                }
            )
    return records


def print_summary(records: list[dict[str, Any]]) -> None:
    """Print a per-class count summary of classification `records` to stdout."""
    counts: dict[str, int] = {}
    for record in records:
        counts[record["predicted_class"]] = counts.get(record["predicted_class"], 0) + 1

    print(f"Classified {len(records)} image(s):")
    for class_label, count in sorted(counts.items()):
        print(f"  {class_label}: {count}")
    for record in records:
        scores = ", ".join(f"{k}={v:.3f}" for k, v in record["scores"].items())
        margin = record["margin"]
        print(f"  {record['path']}: {record['predicted_class']} (margin={margin:.3f}, {scores})")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `clip_classifier.py`."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input-dir", required=True, help="Folder of images to classify.")
    parser.add_argument(
        "--style",
        default="basic",
        help="Prompt style from config/prompts.yaml (basic/industrial/detailed). "
        "Ignored if --use-linear-head is set.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=EDGE_BATCH_SIZE,
        help="Images per encode_image() call (default: 1, the on-device streaming scenario).",
    )
    parser.add_argument("--config", default="config/model_config.yaml")
    parser.add_argument("--prompts", default="config/prompts.yaml")
    parser.add_argument(
        "--use-linear-head",
        default=None,
        metavar="PATH",
        help="Path to a LinearHead .npz (see train_linear_head.py). If set, classify "
        "with it instead of zero-shot cosine similarity to prompts.",
    )
    parser.add_argument(
        "--output", default=None, help="Optional path to write the JSON classification report."
    )
    return parser.parse_args()


def main() -> None:
    """Load config/prompts (or a linear head)/model, classify --input-dir, report results."""
    args = parse_args()

    warning = describe_batch_size_warning(args.batch_size)
    if warning is not None:
        print(f"WARNING: {warning}", file=sys.stderr)

    if args.use_linear_head:
        try:
            linear_head = LinearHead.load(args.use_linear_head)
        except FileNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        mode_metadata = {"mode": "linear_head", "linear_head_path": args.use_linear_head}
    else:
        prompt_manager = PromptManager.from_yaml(args.prompts)
        try:
            prompt_set = prompt_manager.get(args.style)
        except KeyError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        mode_metadata = {"mode": "zero_shot", "style": args.style}

    dataset = ImageFolderDataset(args.input_dir)
    if len(dataset) == 0:
        print(f"Error: no images found under '{args.input_dir}'.", file=sys.stderr)
        sys.exit(1)

    config = load_config(args.config)
    loader = CLIPLoader(config.model, config.memory)
    inference = CLIPInference(loader)

    if args.use_linear_head:
        score_fn = make_linear_head_scorer(linear_head)
    else:
        class_vectors = build_class_reference_vectors(inference, prompt_set.phrases_by_class)
        score_fn = make_zero_shot_scorer(class_vectors)

    records = classify_folder(inference, dataset, args.batch_size, score_fn)
    print_summary(records)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "metadata": {
                "input_dir": args.input_dir,
                "batch_size": args.batch_size,
                "model_name": config.model.name,
                "precision": config.model.precision,
                **mode_metadata,
            },
            "results": records,
        }
        output_path.write_text(json.dumps(report, indent=2))
        print(f"Report written to '{output_path}'.")


if __name__ == "__main__":
    main()
