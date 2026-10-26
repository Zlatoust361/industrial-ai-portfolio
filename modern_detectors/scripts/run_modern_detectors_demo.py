#!/usr/bin/env python3
"""Week 26's whole non-interactive deliverable, one command.

Builds a train/val split from week 25's real synthetic defect dataset
(`classical_cv.defect_images`, itself built on week 05's real
`generate_defect_mask`), resizes it (`modern_detectors.dataset`), trains
a tiny RT-DETR and a from-scratch YOLOv8n on the SAME train split
(`modern_detectors.rt_detr_scratch`/`yolo_scratch`), scores both on the
SAME val split with the SAME IoU=0.5 standard
(`modern_detectors.benchmark`), attempts the real OWL-ViT zero-shot
baseline (`modern_detectors.owlvit_baseline`, honestly reported as
`blocked` if this sandbox's network boundary is hit), and writes one
combined comparison report.

Usage:
    PYTHONPATH=. python3 modern_detectors/scripts/run_modern_detectors_demo.py \\
        [--num-positive 40] [--num-negative 20] [--val-fraction 0.25] \\
        [--rt-detr-steps 60] [--yolo-epochs 6] [--seed 42] [--skip-owlvit]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_MODERN_DETECTORS_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_MODERN_DETECTORS_ROOT.parent))

from classical_cv.defect_images import IMAGE_SIZE, build_demo_dataset  # noqa: E402
from modern_detectors.benchmark import measure_per_image_latency_ms, score_predictions  # noqa: E402
from modern_detectors.dataset import resize_dataset, split_train_val  # noqa: E402
from modern_detectors.owlvit_baseline import OwlVitSummary, run_owlvit_or_blocked  # noqa: E402
from modern_detectors.report import build_report, to_dict  # noqa: E402
from modern_detectors.rt_detr_scratch import (  # noqa: E402
    build_tiny_rt_detr_model,
    predict_rt_detr,
    train_rt_detr,
)
from modern_detectors.yolo_scratch import predict_yolo, train_yolo_from_scratch  # noqa: E402
from src.synthetic.config import load_defects_config  # noqa: E402

DEFAULT_OUTPUT_PATH = "outputs/modern_detectors_demo/report.json"
DEFAULT_DATASET_DIR = "outputs/modern_detectors_demo/yolo_dataset"
DEFAULT_DEFECTS_CONFIG_PATH = "config/defects.yaml"
DEFAULT_NUM_POSITIVE = 40
DEFAULT_NUM_NEGATIVE = 20
DEFAULT_VAL_FRACTION = 0.25
DEFAULT_RT_DETR_STEPS = 200
DEFAULT_RT_DETR_BATCH_SIZE = 4
DEFAULT_YOLO_EPOCHS = 60
"""60, not e.g. 15: empirically, YOLOv8n-scratch collapses to predicting
the whole image as its only box below this (real finding, not a bug --
see `modern_detectors/README.md`'s Honesty note); RT-DETR-tiny does not
show the same collapse at a much shorter 200-step budget."""
DEFAULT_SEED = 42


def run_modern_detectors_demo(
    num_positive: int,
    num_negative: int,
    val_fraction: float,
    rt_detr_steps: int,
    yolo_epochs: int,
    seed: int,
    defects_config_path: str,
    dataset_dir: str,
    skip_owlvit: bool,
) -> dict[str, object]:
    """Run the full week 26 comparison and return a JSON-serializable report."""
    defects_config = load_defects_config(defects_config_path)
    raw_dataset = build_demo_dataset(defects_config, num_positive, num_negative, seed)
    resized = resize_dataset(raw_dataset, IMAGE_SIZE)

    train_images, val_images = split_train_val(resized, val_fraction)

    rt_detr_model = build_tiny_rt_detr_model(seed=seed)
    rt_detr_run = train_rt_detr(
        rt_detr_model,
        train_images,
        num_steps=rt_detr_steps,
        batch_size=DEFAULT_RT_DETR_BATCH_SIZE,
        learning_rate=1e-3,
        seed=seed,
    )
    rt_detr_predictions = predict_rt_detr(rt_detr_run.model, val_images)
    rt_detr_latencies = measure_per_image_latency_ms(
        lambda image: predict_rt_detr(rt_detr_run.model, [image]), val_images
    )
    rt_detr_report = score_predictions(
        "rt_detr_tiny", rt_detr_predictions, val_images, rt_detr_latencies
    )

    yolo_run = train_yolo_from_scratch(
        train_images,
        dataset_dir,
        epochs=yolo_epochs,
        batch_size=DEFAULT_RT_DETR_BATCH_SIZE,
        seed=seed,
    )
    yolo_predictions = predict_yolo(yolo_run.model, val_images)
    yolo_latencies = measure_per_image_latency_ms(
        lambda image: predict_yolo(yolo_run.model, [image]), val_images
    )
    yolo_report = score_predictions("yolov8n_scratch", yolo_predictions, val_images, yolo_latencies)

    if skip_owlvit:
        owlvit_summary = OwlVitSummary(status="blocked", blocked_reason="--skip-owlvit was passed")
    else:
        owlvit_summary = run_owlvit_or_blocked(val_images)

    report = build_report(rt_detr_report, yolo_report, owlvit_summary)
    result = to_dict(report)
    result["rt_detr"]["train_losses"] = list(rt_detr_run.step_losses)
    result["yolo"]["train_seconds"] = yolo_run.train_seconds
    result["num_train_images"] = len(train_images)
    result["num_val_images"] = len(val_images)
    return result


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `run_modern_detectors_demo.py`."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--num-positive", type=int, default=DEFAULT_NUM_POSITIVE)
    parser.add_argument("--num-negative", type=int, default=DEFAULT_NUM_NEGATIVE)
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--rt-detr-steps", type=int, default=DEFAULT_RT_DETR_STEPS)
    parser.add_argument("--yolo-epochs", type=int, default=DEFAULT_YOLO_EPOCHS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--defects-config", default=DEFAULT_DEFECTS_CONFIG_PATH)
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--skip-owlvit",
        action="store_true",
        help="Skip the real OWL-ViT baseline attempt (it is expected to hit this "
        "sandbox's network boundary and be reported as blocked anyway).",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    """Run the demo, write the report, print the headline numbers."""
    args = parse_args()
    report = run_modern_detectors_demo(
        num_positive=args.num_positive,
        num_negative=args.num_negative,
        val_fraction=args.val_fraction,
        rt_detr_steps=args.rt_detr_steps,
        yolo_epochs=args.yolo_epochs,
        seed=args.seed,
        defects_config_path=args.defects_config,
        dataset_dir=args.dataset_dir,
        skip_owlvit=args.skip_owlvit,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    rt_detr, yolo, owlvit = report["rt_detr"], report["yolo"], report["owlvit"]
    print(f"RT-DETR-tiny: AP={rt_detr['average_precision']:.3f} "
          f"mean_latency={rt_detr['mean_latency_ms']:.2f}ms")
    print(f"YOLOv8n-scratch: AP={yolo['average_precision']:.3f} "
          f"mean_latency={yolo['mean_latency_ms']:.2f}ms")
    print(f"OWL-ViT: status={owlvit['status']}")
    print(f"Recommendation: {report['recommendation']}")
    print(f"Full report written to {output_path}")


if __name__ == "__main__":
    main()
