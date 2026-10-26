"""Compare OWL-ViT vs YOLOv8 on one test-set: mAP@0.5, FPS, peak memory.

MUST run directly on the Jetson Orin Nano in **15W** mode -- the standard
sustained operating point, not `MAXN_SUPER` (peak/burst, overstates
sustained throughput) and not `7W` (may throttle detection FPS below what
production traffic needs). Refuses to run in any other mode by default;
`--force` overrides (e.g. to exercise this script's logic on a dev machine).

Test-set layout (YOLO-style; see `test_set.py`):
    test_set/
      images/*.jpg
      labels/*.txt        # "class_id cx cy w h", normalized, one file per image
      classes.txt         # one class name per line, index = class_id

For a meaningful YOLOv8 comparison, `classes.txt` should use the same
class name strings the loaded YOLOv8 checkpoint predicts (see
`yolov8_detector.py`'s caveat).

Usage (on the Jetson, in 15W mode):
    python3 compare_owlvit_yolov8.py \\
        --test-set path/to/test_set \\
        --yolo-model yolov8n.pt \\
        --owlvit-model google/owlvit-base-patch32 \\
        [--output outputs/report.md]

Not runnable in this repo's dev sandbox: no GPU/Jetson/tegrastats, no
network access to download OWL-ViT/YOLOv8 weights (see
course/DECISIONS.md ADR-004). `map_metric.py` (mAP/IoU math) and
`test_set.py` (test-set loading) are pure and unit-tested without any of
that; this script's model-running/monitoring glue has been reviewed
carefully but not executed.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Protocol

from map_metric import Prediction, mean_average_precision
from resource_monitor import PeakRssTracker, TegrastatsRamMonitor
from test_set import load_test_set

REQUIRED_POWER_MODE = "15W"
_NVPMODEL_QUERY_NAME_PATTERN = re.compile(r"NV Power Mode:\s*(\S+)")


class _BackendDetector(Protocol):
    def detect(
        self, image_rgb: Any, class_names: list[str]
    ) -> list[tuple[str, float, tuple[float, float, float, float]]]: ...


def query_current_power_mode() -> str | None:
    """Run `nvpmodel -q` and return the parsed mode name, if available.

    Returns:
        The current power-mode name, or `None` if `nvpmodel` isn't
        present/runnable on this machine (i.e. this isn't running on a Jetson).
    """
    try:
        result = subprocess.run(
            ["nvpmodel", "-q"], capture_output=True, text=True, timeout=5, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    match = _NVPMODEL_QUERY_NAME_PATTERN.search(result.stdout)
    return match.group(1) if match else None


def run_backend(
    name: str,
    detector: _BackendDetector,
    image_paths: list[Path],
    class_names: list[str],
) -> tuple[list[Prediction], float, int | None, float]:
    """Run `detector` over every image, tracking predictions, FPS, and peak memory.

    Args:
        name: Backend name, used only in the printed progress line.
        detector: Something with a `.detect(image_rgb, class_names) ->
            [(class_name, score, box_xyxy), ...]` method.
        image_paths: Test images to run over.
        class_names: Every class to detect, passed through to `detector.detect`.

    Returns:
        `(predictions, fps, peak_system_ram_mb, peak_process_rss_mb)`.
    """
    import numpy as np
    from PIL import Image

    rss_tracker = PeakRssTracker()
    predictions: list[Prediction] = []

    with TegrastatsRamMonitor() as ram_monitor:
        start = time.perf_counter()
        for image_path in image_paths:
            with Image.open(image_path) as image:
                image_rgb = np.array(image.convert("RGB"))
            for class_name, score, box in detector.detect(image_rgb, class_names):
                predictions.append(
                    Prediction(
                        image_id=image_path.name, class_name=class_name, score=score, box=box
                    )
                )
            rss_tracker.sample()
        elapsed = time.perf_counter() - start

    fps = len(image_paths) / elapsed if elapsed > 0 else 0.0
    print(f"[{name}] {len(image_paths)} image(s) in {elapsed:.2f}s ({fps:.2f} FPS)")
    return predictions, fps, ram_monitor.peak_ram_mb(), rss_tracker.peak_mb


def format_report(rows: list[dict[str, Any]], test_set_dir: str, power_mode: str | None) -> str:
    """Render the comparison table as a Markdown report.

    Args:
        rows: One dict per backend: `{"name", "map", "fps", "peak_ram_mb", "peak_rss_mb"}`.
        test_set_dir: Test-set path, recorded for traceability.
        power_mode: The `nvpmodel` mode the run happened under, if known.

    Returns:
        The full Markdown report text.
    """
    lines = [
        "# OWL-ViT vs YOLOv8 comparison",
        "",
        f"- Test set: `{test_set_dir}`",
        f"- nvpmodel power mode: `{power_mode or 'unknown'}`",
        "",
        "| Backend | mAP@0.5 | FPS | Peak system RAM (MB) | Peak process RSS (MB) |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        peak_ram = row["peak_ram_mb"] if row["peak_ram_mb"] is not None else "n/a"
        lines.append(
            f"| {row['name']} | {row['map']:.3f} | {row['fps']:.2f} | "
            f"{peak_ram} | {row['peak_rss_mb']:.1f} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `compare_owlvit_yolov8.py`."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--test-set", required=True, help="Root of a YOLO-style test set.")
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--owlvit-model", default="google/owlvit-base-patch32")
    parser.add_argument("--score-threshold", type=float, default=0.1)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--output", default="outputs/report.md")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip the 15W power-mode gate (results won't be comparable).",
    )
    return parser.parse_args()


def main() -> None:
    """Gate on 15W, run both detectors over the test set, write the comparison report."""
    args = parse_args()

    power_mode = query_current_power_mode()
    if not args.force and power_mode != REQUIRED_POWER_MODE:
        print(
            f"Error: current nvpmodel power mode is {power_mode!r}, not "
            f"{REQUIRED_POWER_MODE!r}. This benchmark represents the standard "
            "sustained operating point, not MAXN Super's peak burst or 7W's "
            "throttled floor -- results at another mode aren't comparable to "
            "future runs. Switch with 'sudo nvpmodel -m 1' (15W), or pass "
            "--force to run anyway (e.g. to exercise this script off-device).",
            file=sys.stderr,
        )
        sys.exit(1)

    image_paths, ground_truths, class_names = load_test_set(args.test_set)
    if not image_paths:
        print(f"Error: no images found under '{args.test_set}/images'.", file=sys.stderr)
        sys.exit(1)

    from owlvit_multi import OwlViTMultiClassDetector
    from yolov8_detector import YOLOv8Detector

    backends: list[tuple[str, _BackendDetector]] = [
        ("OWL-ViT", OwlViTMultiClassDetector(args.owlvit_model, args.score_threshold)),
        ("YOLOv8", YOLOv8Detector(args.yolo_model, args.score_threshold)),
    ]

    rows = []
    for name, detector in backends:
        predictions, fps, peak_ram_mb, peak_rss_mb = run_backend(
            name, detector, image_paths, class_names
        )
        map_result = mean_average_precision(predictions, ground_truths, args.iou_threshold)
        rows.append(
            {
                "name": name,
                "map": map_result["mAP"],
                "fps": fps,
                "peak_ram_mb": peak_ram_mb,
                "peak_rss_mb": peak_rss_mb,
            }
        )

    report = format_report(rows, args.test_set, power_mode)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    print(report)
    print(f"Report written to '{output_path}'.")


if __name__ == "__main__":
    main()
