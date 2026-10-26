"""Aggregation + Markdown rendering for `08_profile.py` (week 09, day 4).

Pure JSON-in, Markdown-out -- unlike every other `quantization/utils/*.py`
module this week, nothing here touches `tensorrt`/`pycuda`/`torch` at all.
That's exactly why `08_profile.py` itself isn't Jetson-gated: aggregating
already-written JSON reports from `06_trt_inference.py`/
`07_evaluate_quality.py` is portable, the same way week 07's
`compare_dr_branches.py` could compute real statistics/plots from
already-computed `detections.json` without needing `ultralytics` itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def discover_trt_inference_reports(inference_dir: Path) -> list[dict[str, Any]]:
    """Load every JSON report `06_trt_inference.py` wrote under `inference_dir`.

    Args:
        inference_dir: `<outputs_dir>/trt_inference/` -- one JSON file per
            engine/power-mode combination `06_trt_inference.py` was run for.

    Returns:
        Parsed report dicts, sorted by filename for determinism.

    Raises:
        FileNotFoundError: If `inference_dir` doesn't exist or has no `.json` files.
    """
    if not inference_dir.is_dir():
        raise FileNotFoundError(
            f"TensorRT inference directory not found: {inference_dir}. Run "
            "06_trt_inference.py (once per engine/power mode) first."
        )
    paths = sorted(inference_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(
            f"No TensorRT inference reports found under '{inference_dir}'. Run "
            "06_trt_inference.py (once per engine/power mode) first."
        )
    return [json.loads(path.read_text()) for path in paths]


def load_optional_json(path: Path) -> dict[str, Any] | None:
    """Load `path` as JSON if it exists, else `None` -- for the optional inputs.

    Args:
        path: A JSON file that may or may not exist yet (the baseline/
            quality reports are useful additions to the summary, but
            08_profile.py's core job -- the speedup table -- doesn't need
            them).

    Returns:
        The parsed JSON, or `None` if `path` doesn't exist.
    """
    return json.loads(path.read_text()) if path.is_file() else None


def power_mode_name(report: dict[str, Any]) -> str:
    """The report's `nvpmodel` mode name, or `"unknown"` if it wasn't recorded.

    Args:
        report: One `06_trt_inference.py` report dict.

    Returns:
        `metadata.power_mode.mode_name`, or `"unknown"` if `power_mode` is
        `None`/missing (e.g. `nvpmodel` wasn't queryable when the run happened).
    """
    power_mode = report.get("metadata", {}).get("power_mode") or {}
    return power_mode.get("mode_name") or "unknown"


def group_by_power_mode(reports: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    """Group reports as `{power_mode_name: {label: report}}`.

    Args:
        reports: From `discover_trt_inference_reports`.

    Returns:
        E.g. `{"15W": {"fp16": {...}, "int4": {...}}, "MAXN_SUPER": {...}}`.
        If two reports share the same `(power_mode, label)` pair (e.g. `06`
        was re-run), the later one in `reports`' order wins -- callers pass
        an already-sorted-by-filename list, so this is deterministic, not
        "whichever happened to run last".
    """
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for report in reports:
        label = report["metadata"]["label"]
        grouped.setdefault(power_mode_name(report), {})[label] = report
    return grouped


@dataclass(frozen=True)
class SpeedupRow:
    """One power mode's FP16-vs-INT4 speedup, or a note explaining why it's incomplete.

    Attributes:
        power_mode: `nvpmodel` mode name (or `"unknown"`).
        fp16/int4 latency fields: `None` if that engine's report is missing
            for this mode (see `note`).
        speedup_mean/speedup_p95: `fp16_*_ms / int4_*_ms` -- `None` if
            either side is missing, or `int4`'s latency is `0`.
        note: Set (and every numeric field left `None`) when this mode is
            missing either engine's report -- computing a "speedup" from
            only one side would be meaningless, not just incomplete.
    """

    power_mode: str
    fp16_mean_ms: float | None
    int4_mean_ms: float | None
    speedup_mean: float | None
    fp16_p95_ms: float | None
    int4_p95_ms: float | None
    speedup_p95: float | None
    fp16_throughput_ips: float | None
    int4_throughput_ips: float | None
    note: str | None


def compute_speedup_rows(grouped: dict[str, dict[str, dict[str, Any]]]) -> list[SpeedupRow]:
    """Compute one `SpeedupRow` per power mode observed in `grouped`.

    Args:
        grouped: From `group_by_power_mode`.

    Returns:
        One row per power mode, sorted by mode name.
    """
    rows: list[SpeedupRow] = []
    for mode in sorted(grouped):
        pair = grouped[mode]
        fp16, int4 = pair.get("fp16"), pair.get("int4")
        if fp16 is None or int4 is None:
            missing = "fp16" if fp16 is None else "int4"
            rows.append(
                SpeedupRow(
                    mode,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    f"missing '{missing}' report for this mode",
                )
            )
            continue

        fp16_lat, int4_lat = fp16["latency"], int4["latency"]
        speedup_mean = (
            fp16_lat["mean_ms"] / int4_lat["mean_ms"] if int4_lat["mean_ms"] > 0 else None
        )
        speedup_p95 = fp16_lat["p95_ms"] / int4_lat["p95_ms"] if int4_lat["p95_ms"] > 0 else None
        rows.append(
            SpeedupRow(
                power_mode=mode,
                fp16_mean_ms=fp16_lat["mean_ms"],
                int4_mean_ms=int4_lat["mean_ms"],
                speedup_mean=speedup_mean,
                fp16_p95_ms=fp16_lat["p95_ms"],
                int4_p95_ms=int4_lat["p95_ms"],
                speedup_p95=speedup_p95,
                fp16_throughput_ips=fp16_lat["throughput_ips"],
                int4_throughput_ips=int4_lat["throughput_ips"],
                note=None,
            )
        )
    return rows


def aggregate_peak_temperatures(reports: list[dict[str, Any]]) -> dict[str, float]:
    """Peak temperature per sensor across every report in this test cycle.

    Args:
        reports: From `discover_trt_inference_reports` -- each has a
            `temperatures.max_temps_c` mapping from `06_trt_inference.py`.

    Returns:
        `{sensor: peak_celsius}`, across every run. Empty if no report has
        any temperature readings (e.g. no `tegrastats` binary on the host
        that produced them -- see `06_trt_inference.py`'s own honesty note).
    """
    peaks: dict[str, float] = {}
    for report in reports:
        for sensor, value in report.get("temperatures", {}).get("max_temps_c", {}).items():
            peaks[sensor] = max(peaks.get(sensor, value), value)
    return peaks


def is_thermal_warning(peaks: dict[str, float], threshold_c: float) -> bool:
    """Whether any sensor's peak (across the whole test cycle) is at/above `threshold_c`.

    Args:
        peaks: From `aggregate_peak_temperatures`.
        threshold_c: `evaluation.throttle_temp_c` -- same configurable-proxy
            caveat as week 08's `src/api/metrics.py::THROTTLE_TEMP_C`.

    Returns:
        `True` if any sensor's peak reading is `>= threshold_c`.
    """
    return any(value >= threshold_c for value in peaks.values())


def _format(value: float | None, precision: int = 2, suffix: str = "") -> str:
    """Render a possibly-`None` float for a Markdown table cell."""
    return "—" if value is None else f"{value:.{precision}f}{suffix}"


def render_summary_markdown(
    *,
    reports: list[dict[str, Any]],
    grouped: dict[str, dict[str, dict[str, Any]]],
    speedup_rows: list[SpeedupRow],
    quality_report: dict[str, Any] | None,
    baseline_report: dict[str, Any] | None,
    peak_temps_c: dict[str, float],
    throttle_temp_c: float,
    generated_at: str,
) -> str:
    """Render the full `summary.md` from every input `08_profile.py` gathered.

    Args:
        reports: From `discover_trt_inference_reports`.
        grouped: From `group_by_power_mode`.
        speedup_rows: From `compute_speedup_rows`.
        quality_report: From `load_optional_json` on `07`'s output, or `None`.
        baseline_report: From `load_optional_json` on `01`'s output, or `None`.
        peak_temps_c: From `aggregate_peak_temperatures`.
        throttle_temp_c: `evaluation.throttle_temp_c`.
        generated_at: An ISO-8601 timestamp string, stamped by the caller
            (this module makes no `datetime.now()` calls itself, so it
            stays trivially unit-testable with a fixed value).

    Returns:
        The full Markdown report text.
    """
    lines: list[str] = [
        "# Week 09 — FP16 vs INT4 TensorRT engine comparison",
        "",
        f"Generated: {generated_at}",
        f"TensorRT inference reports aggregated: {len(reports)} "
        f"({', '.join(sorted({power_mode_name(r) for r in reports}))})",
        "",
        "## Speedup: FP16 → INT4",
        "",
        "| Power mode | FP16 mean (ms) | INT4 mean (ms) | Speedup (mean) | "
        "FP16 p95 (ms) | INT4 p95 (ms) | Speedup (p95) |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in speedup_rows:
        if row.note is not None:
            lines.append(f"| {row.power_mode} | {row.note} | | | | | |")
            continue
        lines.append(
            f"| {row.power_mode} | {_format(row.fp16_mean_ms)} | {_format(row.int4_mean_ms)} | "
            f"{_format(row.speedup_mean, suffix='x')} | {_format(row.fp16_p95_ms)} | "
            f"{_format(row.int4_p95_ms)} | {_format(row.speedup_p95, suffix='x')} |"
        )

    lines += [
        "",
        "## Trade-off: latency vs quality",
        "",
        "| Engine | Power mode | Batch | Mean latency (ms) | Throughput (img/s) | "
        "Mean cosine similarity to FP16 | Quality gate |",
        "|---|---|---|---|---|---|---|",
    ]
    quality_similarity = quality_report["mean_cosine_similarity"] if quality_report else None
    quality_passed = quality_report["passed"] if quality_report else None
    for mode in sorted(grouped):
        for label in ("fp16", "int4"):
            report = grouped[mode].get(label)
            if report is None:
                continue
            latency = report["latency"]
            similarity = (
                "1.0000 (reference)"
                if label == "fp16"
                else _format(quality_similarity, precision=4)
            )
            if label == "fp16":
                gate = "—"
            elif quality_passed is None:
                gate = "—"
            else:
                gate = "PASSED" if quality_passed else "FAILED"
            lines.append(
                f"| {label} | {mode} | {latency['batch_size']} | {_format(latency['mean_ms'])} | "
                f"{_format(latency['throughput_ips'])} | {similarity} | {gate} |"
            )
    if quality_report is None:
        lines.append("")
        lines.append(
            "_No `07_evaluate_quality.py` report found -- quality column left blank. "
            "Run `07_evaluate_quality.py` to fill it in._"
        )

    lines += [
        "",
        "## Temperature / throttling",
        "",
        f"Peak temperatures observed across all {len(reports)} runs in this test cycle "
        f"(thermal-warning threshold: {throttle_temp_c:.1f}°C — a configurable proxy, "
        "not a real DVFS trip point; see `src/api/metrics.py::THROTTLE_TEMP_C`'s "
        "docstring in the root repo for the same caveat):",
        "",
    ]
    if not peak_temps_c:
        lines.append(
            "_No `tegrastats` readings in any report -- either `tegrastats` wasn't "
            "available on the host that produced them, or none were sampled._"
        )
    else:
        lines.append("| Sensor | Peak temperature (°C) | Warning |")
        lines.append("|---|---|---|")
        for sensor in sorted(peak_temps_c):
            value = peak_temps_c[sensor]
            warning = "⚠️ at/above threshold" if value >= throttle_temp_c else "OK"
            lines.append(f"| {sensor} | {value:.1f} | {warning} |")
        lines.append("")
        overall = "YES" if is_thermal_warning(peak_temps_c, throttle_temp_c) else "no"
        lines.append(f"**Thermal warning triggered:** {overall}")

    lines += [
        "",
        "### Per-run breakdown",
        "",
        "| Label | Power mode | Mean temps (°C) | Max temps (°C) |",
        "|---|---|---|---|",
    ]
    for report in reports:
        label = report["metadata"]["label"]
        mode = power_mode_name(report)
        temps = report.get("temperatures", {})
        mean_temps = sorted(temps.get("mean_temps_c", {}).items())
        max_temps = sorted(temps.get("max_temps_c", {}).items())
        mean_str = ", ".join(f"{s}={v:.1f}" for s, v in mean_temps) or "—"
        max_str = ", ".join(f"{s}={v:.1f}" for s, v in max_temps) or "—"
        lines.append(f"| {label} | {mode} | {mean_str} | {max_str} |")

    lines += ["", "## Inputs"]
    if baseline_report is not None:
        precision = baseline_report.get("metadata", {}).get("precision", "?")
        baseline_batch1 = next(
            (r for r in baseline_report.get("results", []) if r.get("batch_size") == 1), None
        )
        if baseline_batch1 is not None:
            lines.append(
                f"- Baseline (PyTorch {precision}, day 1): mean latency @ batch=1: "
                f"{baseline_batch1['mean_ms']:.2f} ms (`outputs/baseline/baseline_results.json`)."
            )
        else:
            lines.append("- Baseline report found (`outputs/baseline/baseline_results.json`).")
    else:
        lines.append(
            "- No `01_baseline_inference.py` report found -- baseline comparison omitted."
        )
    lines.append(
        f"- TensorRT inference reports: {len(reports)} found under `outputs/trt_inference/`."
    )
    if quality_report is not None:
        lines.append(
            f"- Quality report: `outputs/quality/quality_report.json` "
            f"({quality_report['num_images']} images compared)."
        )
    else:
        lines.append("- No `07_evaluate_quality.py` report found.")

    return "\n".join(lines) + "\n"
