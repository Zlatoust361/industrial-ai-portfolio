"""PR/ROC curve plotting (week 07, day 4).

Lazily imports `matplotlib` (forced to the non-interactive `"Agg"` backend
-- this runs from CLI scripts with no display, and importing a GUI backend
without one raises) so importing this module costs nothing if
`matplotlib` isn't installed, consistent with how heavy/optional
dependencies are handled everywhere else in this repo.
"""

from __future__ import annotations

from pathlib import Path


def plot_pr_curves(
    curves: dict[str, tuple[list[float], list[float]]],
    output_path: str | Path,
    title: str = "Precision-Recall curve",
) -> Path:
    """Plot one or more precision-recall curves on a single figure.

    Args:
        curves: Series label (e.g. `"no_dr"`) -> `(recalls, precisions)`,
            as returned by `src.evaluation.detection_metrics.precision_recall_curve`.
        output_path: Where to save the figure (`.png`).
        title: Plot title.

    Returns:
        `output_path`, as a `Path`.

    Raises:
        ValueError: If `curves` is empty.
    """
    if not curves:
        raise ValueError("plot_pr_curves needs at least one curve.")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(figsize=(6, 6))
    for label, (recalls, precisions) in curves.items():
        axes.plot(recalls, precisions, marker=".", label=label)
    axes.set_xlabel("Recall")
    axes.set_ylabel("Precision")
    axes.set_xlim(0.0, 1.0)
    axes.set_ylim(0.0, 1.05)
    axes.set_title(title)
    axes.legend()
    axes.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def plot_roc_curves(
    curves: dict[str, tuple[list[float], list[float], float]],
    output_path: str | Path,
    title: str = "ROC curve",
) -> Path:
    """Plot one or more ROC curves (with AUC in the legend) on a single figure.

    Args:
        curves: Series label -> `(fpr, tpr, auc)`, `fpr`/`tpr` as returned
            by `src.evaluation.detection_metrics.roc_curve`, `auc` from
            `roc_auc_score`.
        output_path: Where to save the figure (`.png`).
        title: Plot title.

    Returns:
        `output_path`, as a `Path`.

    Raises:
        ValueError: If `curves` is empty.
    """
    if not curves:
        raise ValueError("plot_roc_curves needs at least one curve.")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(figsize=(6, 6))
    for label, (fpr, tpr, auc) in curves.items():
        axes.plot(fpr, tpr, marker=".", label=f"{label} (AUC={auc:.3f})")
    axes.plot([0, 1], [0, 1], linestyle="--", color="gray", label="chance")
    axes.set_xlabel("False positive rate")
    axes.set_ylabel("True positive rate")
    axes.set_xlim(0.0, 1.0)
    axes.set_ylim(0.0, 1.05)
    axes.set_title(title)
    axes.legend()
    axes.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path
