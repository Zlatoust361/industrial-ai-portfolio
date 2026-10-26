#!/usr/bin/env python3
"""Week 28's whole non-interactive deliverable, one command.

Trains a wide "teacher" and a narrow "student" (both real `TinyCNN`s,
see `distillation.models`) on the SAME binary defect-classification
split built from week 25's real synthetic images
(`distillation.dataset`, itself built on week 05's real mask generator).
Compares, with the same real measurement methodology
(`distillation.benchmark.measure_row`): the student trained alone vs.
with real knowledge distillation from the teacher
(`distillation.distillation`), a structurally pruned teacher
(`distillation.pruning`), the best architecture found by a small basic
NAS search (`distillation.nas`), and a real INT8-quantized teacher
(`distillation.quantization_int8`). Writes one combined report.

Usage:
    PYTHONPATH=. python3 distillation/scripts/run_distillation_demo.py \\
        [--num-positive 40] [--num-negative 20] [--val-fraction 0.25] \\
        [--teacher-steps 500] [--student-steps 1200] [--nas-steps 400] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_DISTILLATION_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_DISTILLATION_ROOT.parent))

from distillation.benchmark import measure_row  # noqa: E402
from distillation.dataset import build_train_val_split  # noqa: E402
from distillation.distillation import train_student_with_distillation  # noqa: E402
from distillation.models import STUDENT_WIDTHS, TEACHER_WIDTHS, TinyCNN, count_parameters  # noqa: E402
from distillation.nas import run_nas_search  # noqa: E402
from distillation.pruning import prune_structured  # noqa: E402
from distillation.quantization_int8 import quantize_int8  # noqa: E402
from distillation.report import build_report, to_dict  # noqa: E402
from distillation.training import train_classifier  # noqa: E402
from src.synthetic.config import load_defects_config  # noqa: E402

DEFAULT_OUTPUT_PATH = "outputs/distillation_demo/report.json"
DEFAULT_DEFECTS_CONFIG_PATH = "config/defects.yaml"
DEFAULT_NUM_POSITIVE = 40
DEFAULT_NUM_NEGATIVE = 20
DEFAULT_VAL_FRACTION = 0.25
DEFAULT_TEACHER_STEPS = 500
DEFAULT_STUDENT_STEPS = 1200
"""Longer than `DEFAULT_TEACHER_STEPS`: empirically, the narrow student
architecture needs ~1000 steps to escape the training set's class-prior
baseline on this task, where the wider teacher does it in ~400 -- see
`distillation/README.md`'s Honesty note."""
DEFAULT_NAS_STEPS = 400
DEFAULT_BATCH_SIZE = 8
DEFAULT_LEARNING_RATE = 2e-3
DEFAULT_PRUNE_KEEP_FRACTION = 0.5
DEFAULT_SEED = 42


def run_distillation_demo(
    num_positive: int,
    num_negative: int,
    val_fraction: float,
    teacher_steps: int,
    student_steps: int,
    nas_steps: int,
    defects_config_path: str,
    seed: int,
) -> dict[str, object]:
    """Run the full week 28 comparison and return a JSON-serializable report."""
    defects_config = load_defects_config(defects_config_path)
    train, val = build_train_val_split(
        defects_config, num_positive, num_negative, val_fraction, seed
    )

    teacher = TinyCNN(TEACHER_WIDTHS)
    train_classifier(teacher, train, teacher_steps, DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE, seed)
    teacher_row = measure_row("teacher", teacher, val)

    student_plain = TinyCNN(STUDENT_WIDTHS)
    train_classifier(
        student_plain, train, student_steps, DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE, seed
    )
    student_plain_row = measure_row("student_plain", student_plain, val)

    student_distilled = TinyCNN(STUDENT_WIDTHS)
    train_student_with_distillation(
        student_distilled, teacher, train, student_steps, DEFAULT_BATCH_SIZE,
        DEFAULT_LEARNING_RATE, seed,
    )
    student_distilled_row = measure_row("student_distilled", student_distilled, val)

    pruned = prune_structured(teacher, keep_fraction=DEFAULT_PRUNE_KEEP_FRACTION)
    pruned_row = measure_row("pruned_50pct", pruned, val)

    nas_result = run_nas_search(
        train, val, max_parameters=count_parameters(student_plain), num_steps=nas_steps,
        batch_size=DEFAULT_BATCH_SIZE, learning_rate=DEFAULT_LEARNING_RATE, seed=seed,
    )
    nas_best_model = TinyCNN(nas_result.best.widths)
    train_classifier(
        nas_best_model, train, nas_steps, DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE, seed
    )
    nas_row = measure_row("nas_best", nas_best_model, val)

    quantized = quantize_int8(teacher, train.images[: min(16, len(train.images))])
    quantized_row = measure_row("int8_quantized", quantized, val)

    report = build_report(
        [
            teacher_row,
            student_plain_row,
            student_distilled_row,
            pruned_row,
            nas_row,
            quantized_row,
        ]
    )
    result = to_dict(report)
    result["nas_search_space"] = [
        {"widths": list(c.widths), "num_parameters": c.num_parameters, "accuracy": c.accuracy}
        for c in nas_result.all_candidates
    ]
    result["nas_best_widths"] = list(nas_result.best.widths)
    return result


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `run_distillation_demo.py`."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--num-positive", type=int, default=DEFAULT_NUM_POSITIVE)
    parser.add_argument("--num-negative", type=int, default=DEFAULT_NUM_NEGATIVE)
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--teacher-steps", type=int, default=DEFAULT_TEACHER_STEPS)
    parser.add_argument("--student-steps", type=int, default=DEFAULT_STUDENT_STEPS)
    parser.add_argument("--nas-steps", type=int, default=DEFAULT_NAS_STEPS)
    parser.add_argument("--defects-config", default=DEFAULT_DEFECTS_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    """Run the demo, write the report, print the headline numbers."""
    args = parse_args()
    report = run_distillation_demo(
        num_positive=args.num_positive,
        num_negative=args.num_negative,
        val_fraction=args.val_fraction,
        teacher_steps=args.teacher_steps,
        student_steps=args.student_steps,
        nas_steps=args.nas_steps,
        defects_config_path=args.defects_config,
        seed=args.seed,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for row in report["rows"]:
        print(
            f"{row['technique']}: acc={row['accuracy']:.3f} "
            f"params={row['num_parameters']} size={row['size_kb']:.1f}KiB "
            f"latency={row['latency_ms']:.3f}ms"
        )
    print(f"NAS best architecture: {report['nas_best_widths']}")
    print(f"Recommendation: {report['recommendation']}")
    print(f"Full report written to {output_path}")


if __name__ == "__main__":
    main()
