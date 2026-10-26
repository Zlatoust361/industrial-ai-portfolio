"""CLI: train a thin logistic-regression head on frozen CLIP embeddings.

Encodes every image under `--train-dir` (a `class_label/...` layout, same
as `optimize_prompts.py`'s validation sets) with the frozen CLIP image
encoder, fits `sklearn.linear_model.LogisticRegression` on those
embeddings, and saves just the resulting weight vector + bias as a
`LinearHead` `.npz` -- kilobytes, not the whole sklearn estimator. Use it
at inference time via `scripts/clip_classifier.py --use-linear-head PATH`.

Usage:
    PYTHONPATH=. python scripts/train_linear_head.py --train-dir path/to/train \\
        [--test-dir path/to/held_out] [--config config/model_config.yaml] \\
        [--positive-class defect] [--negative-class ok] \\
        [--output outputs/week02/linear_head.npz] [--batch-size 8] \\
        [--max-iter 1000] [--c 1.0]

`--train-dir`/`--test-dir` must contain one subfolder per class (default
`ok/` and `defect/`).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from src.config import load_config
from src.data.dataset import LabeledImageFolderDataset, iter_batches
from src.evaluation.prompt_optimization import compute_auc, labels_to_binary
from src.models.clip import CLIPInference, CLIPLoader
from src.models.linear_head import LinearHead, check_linear_head_file_size

DEFAULT_OUTPUT = "outputs/week02/linear_head.npz"


def encode_dataset_images(
    inference: CLIPInference, dataset: LabeledImageFolderDataset, batch_size: int
) -> np.ndarray:
    """Encode every image in `dataset` into CLIP embeddings, batched.

    Args:
        inference: A loaded (or lazily-loading) `CLIPInference`.
        dataset: Labeled images to encode.
        batch_size: Images per `encode_image` call.

    Returns:
        L2-normalized embeddings, shape `(len(dataset), embedding_dim)`, in dataset order.
    """
    batches = []
    for batch in iter_batches(dataset.samples, batch_size):
        images = [sample.load() for sample in batch]
        batches.append(inference.encode_image(images).vectors)
    return np.concatenate(batches, axis=0)


def report_metrics(
    name: str, head: LinearHead, embeddings: np.ndarray, labels_bin: np.ndarray
) -> None:
    """Print AUC + accuracy for `head` on an already-encoded (embeddings, labels) set."""
    probabilities = head.predict_proba(embeddings)
    predictions = head.predict(embeddings)
    positive_labels = [
        head.positive_class if label == 1 else head.negative_class for label in labels_bin
    ]
    correct = sum(p == t for p, t in zip(predictions, positive_labels, strict=True))
    accuracy = correct / len(labels_bin)
    auc = compute_auc(probabilities, labels_bin)
    print(f"{name}: AUC={auc:.3f}, accuracy={accuracy:.3f} (n={len(labels_bin)})")


def load_labeled_dataset_or_exit(path: str, class_labels: list[str]) -> LabeledImageFolderDataset:
    """Load a `LabeledImageFolderDataset`, exiting with a clean error message on failure."""
    try:
        dataset = LabeledImageFolderDataset(path, class_labels=class_labels)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    counts = {label: 0 for label in class_labels}
    for sample in dataset.samples:
        counts[sample.label] += 1
    empty_classes = [label for label, count in counts.items() if count == 0]
    if empty_classes:
        print(
            f"Error: no images found for class(es) {empty_classes} under '{path}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    return dataset


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `train_linear_head.py`."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--train-dir", required=True, help="Folder with one subfolder per class.")
    parser.add_argument(
        "--test-dir", default=None, help="Optional held-out folder for evaluation after training."
    )
    parser.add_argument("--config", default="config/model_config.yaml")
    parser.add_argument("--positive-class", default="defect")
    parser.add_argument("--negative-class", default="ok")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Images per encode_image() call; defaults to inference.batch_size from config "
        "-- training is offline bulk processing, not the edge streaming scenario.",
    )
    parser.add_argument(
        "--max-iter", type=int, default=1000, help="sklearn LogisticRegression max_iter."
    )
    parser.add_argument(
        "--c",
        type=float,
        default=1.0,
        help="sklearn LogisticRegression C (inverse regularization).",
    )
    return parser.parse_args()


def main() -> None:
    """Encode --train-dir with CLIP, fit LogisticRegression, save a LinearHead."""
    from sklearn.linear_model import LogisticRegression

    args = parse_args()
    class_labels = [args.negative_class, args.positive_class]

    config = load_config(args.config)
    batch_size = args.batch_size if args.batch_size is not None else config.inference.batch_size

    train_dataset = load_labeled_dataset_or_exit(args.train_dir, class_labels)

    loader = CLIPLoader(config.model, config.memory)
    inference = CLIPInference(loader)

    print(f"Encoding {len(train_dataset)} training image(s)...")
    train_embeddings = encode_dataset_images(inference, train_dataset, batch_size)
    train_labels_bin = labels_to_binary(
        [sample.label for sample in train_dataset.samples], positive_label=args.positive_class
    )

    model = LogisticRegression(max_iter=args.max_iter, C=args.c)
    model.fit(train_embeddings, train_labels_bin)

    head = LinearHead(
        weights=model.coef_[0],
        bias=float(model.intercept_[0]),
        positive_class=args.positive_class,
        negative_class=args.negative_class,
    )
    head.save(args.output)
    size_kb = check_linear_head_file_size(args.output)
    print(f"Linear head saved to '{args.output}' ({size_kb:.2f} KB) -- confirmed kilobyte-sized.")

    report_metrics("Train", head, train_embeddings, train_labels_bin)

    if args.test_dir:
        test_dataset = load_labeled_dataset_or_exit(args.test_dir, class_labels)
        test_embeddings = encode_dataset_images(inference, test_dataset, batch_size)
        test_labels_bin = labels_to_binary(
            [sample.label for sample in test_dataset.samples], positive_label=args.positive_class
        )
        report_metrics("Test", head, test_embeddings, test_labels_bin)


if __name__ == "__main__":
    main()
