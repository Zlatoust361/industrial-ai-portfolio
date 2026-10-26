"""Renders the final Markdown inspection report from every upstream stage's artifacts.

Pure string templating over already-computed dicts -- no I/O, no model
dependency. `07_generate_report.py` reads `outputs/env_report.json`,
`outputs/interpretations.json` and `outputs/metrics/quality_gate.json` and
passes their parsed contents straight through to `render_report` below.
"""

from __future__ import annotations


def render_report(
    title: str,
    env_report: dict[str, object],
    interpret_summary: dict[str, object],
    quality_gate: dict[str, object],
    verdicts: list[dict[str, object]],
    max_defects_listed: int = 20,
) -> str:
    """Render the full pipeline run as one Markdown report.

    Args:
        title: Report heading, from `config.report.title`.
        env_report: Parsed `outputs/env_report.json` (see
            `utils.environment.EnvironmentReport`).
        interpret_summary: The `summary` half of `utils.interpret.interpret_detections`'s
            return value (parsed from `outputs/interpretations.json`).
        quality_gate: Parsed `outputs/metrics/quality_gate.json` (see
            `utils.quality_gate.QualityGateResult.to_dict`).
        verdicts: The `verdicts` list from `outputs/interpretations.json`
            (each a dict with `"image"`/`"defect_score"`/`"verdict"`).
        max_defects_listed: Cap on how many flagged images get their own
            table row -- large batches would otherwise make the report
            unreadable. The count beyond this cap is always stated
            explicitly (never silently dropped).

    Returns:
        The full report as a Markdown string.
    """
    lines = [f"# {title}", "", "## Environment", ""]
    lines.append(f"- Jetson: `{env_report.get('is_jetson')}`")
    if env_report.get("is_jetson"):
        lines.append(f"- L4T version: `{env_report.get('l4t_version')}`")
        lines.append(f"- JetPack generation: `{env_report.get('jetpack_generation')}`")
    available_mb = float(env_report.get("available_memory_mb") or 0.0)
    lines.append(f"- Available memory: {available_mb:.1f} MB")
    lines.append(f"- Power mode: `{env_report.get('power_mode') or 'unknown'}`")

    total = interpret_summary.get("total", 0)
    defect_count = interpret_summary.get("defect_count", 0)
    defect_rate = float(interpret_summary.get("defect_rate") or 0.0)
    lines += [
        "",
        "## Summary",
        "",
        f"- Total images: {total}",
        f"- Defects: {defect_count}",
        f"- Defect rate: {defect_rate:.3f}",
    ]

    status = quality_gate.get("status", "UNKNOWN")
    thresholds = quality_gate.get("thresholds", {}) or {}
    lines += [
        "",
        "## Quality gate",
        "",
        f"**Status: {status}**",
        f"- Thresholds: GREEN < {thresholds.get('green_max_defect_rate')}, "
        f"YELLOW < {thresholds.get('yellow_max_defect_rate')}, otherwise RED",
    ]

    defects = [entry for entry in verdicts if entry.get("verdict") == "defect"]
    lines += ["", f"## Flagged images ({len(defects)})", ""]
    if defects:
        lines += ["| Image | Defect score |", "|---|---|"]
        for entry in defects[:max_defects_listed]:
            score = float(entry.get("defect_score") or 0.0)
            lines.append(f"| {entry.get('image')} | {score:.3f} |")
        remaining = len(defects) - max_defects_listed
        if remaining > 0:
            lines.append(f"| ...{remaining} more not shown | |")
    else:
        lines.append("None.")
    lines.append("")

    return "\n".join(lines)
