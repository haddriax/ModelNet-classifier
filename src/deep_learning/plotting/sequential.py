"""Sequential training plotting utilities.

Generates comparison plots from sequential training results JSON.
All functions read from the standardized results format and produce PNG files.
"""

import json
from pathlib import Path

from ._utils import _get_model_colors


def plot_sequential_results(results_path: Path, output_dir: Path | None = None) -> None:
    """Generate tailored plots for a sequential training experiment.

    Produces four figures suited to the one-run-per-model structure:

    * ``sequential_model_comparison.png`` — grouped bar chart of accuracy,
      F1, precision and recall for each model.
    * ``sequential_per_class_accuracy.png`` — heatmap of per-class accuracy
      (models × classes).
    * ``sequential_per_class_f1.png`` — same heatmap for per-class F1.
    * ``sequential_training_efficiency.png`` — dual-axis epochs/time chart.

    Args:
        results_path: Path to ``sequential_results.json``.
        output_dir: Directory to save plots (default: same directory as JSON).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if output_dir is None:
        output_dir = results_path.parent

    with open(results_path) as f:
        data = json.load(f)

    runs = [r for r in data["runs"] if r.get("status") == "completed"]

    if not runs:
        print("No completed runs to plot")
        return

    plot_model_comparison(runs, output_dir, plt)
    plot_per_class_heatmap(runs, output_dir, plt, metric_key="per_class_accuracies",
                           title="Per-Class Accuracy by Model",
                           filename="sequential_per_class_accuracy.png",
                           colorbar_label="Class Accuracy (%)")
    plot_per_class_heatmap(runs, output_dir, plt, metric_key="per_class_f1",
                           title="Per-Class F1 Score by Model",
                           filename="sequential_per_class_f1.png",
                           colorbar_label="Class F1 (%)")
    plot_training_efficiency(runs, output_dir, plt)

    print(f"Plots saved to: {output_dir}")


def plot_model_comparison(runs: list[dict], output_dir: Path, plt) -> None:
    """Grouped bar chart comparing accuracy, F1, precision and recall per model.

    Each model gets a group of four bars (one per metric), making it easy to
    compare overall performance and the precision/recall balance at a glance.

    Args:
        runs: List of completed run result dicts.
        output_dir: Directory to save the plot.
        plt: The matplotlib.pyplot module (passed by caller).
    """
    import numpy as np

    model_colors = _get_model_colors()

    # Metric definitions: (result-dict key, display label, bar color)
    metrics_spec = [
        ("best_test_acc",  "Accuracy",  "#2c3e50"),
        ("macro_f1",       "F1",        "#8e44ad"),
        ("macro_precision","Precision", "#16a085"),
        ("macro_recall",   "Recall",    "#d35400"),
    ]

    models = [r["config"]["model"] for r in runs]
    n_models = len(models)
    n_metrics = len(metrics_spec)
    x = np.arange(n_models)
    width = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=(max(8, n_models * 1.8), 6))

    for i, (key, label, color) in enumerate(metrics_spec):
        values = [r["metrics"][key] * 100 for r in runs]
        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=label, color=color, alpha=0.85)

        # Value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.4,
                f"{val:.1f}",
                ha="center", va="bottom", fontsize=7,
            )

    ax.set_xlabel("Model Architecture")
    ax.set_ylabel("Score (%)")
    ax.set_title("Sequential Training: Model Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim(0, 110)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "sequential_model_comparison.png", dpi=150)
    plt.close()


def plot_per_class_heatmap(
    runs: list[dict],
    output_dir: Path,
    plt,
    *,
    metric_key: str,
    title: str,
    filename: str,
    colorbar_label: str,
) -> None:
    """Heatmap of a per-class metric: models (rows) × classes (columns).

    Cells are annotated with the value to one decimal place. The color map
    matches the existing ``plot_model_heatmap()`` (``YlOrRd``).

    Args:
        runs: List of completed run result dicts.
        output_dir: Directory to save the plot.
        plt: The matplotlib.pyplot module (passed by caller).
        metric_key: Key inside ``metrics`` dict, e.g. ``"per_class_accuracies"``
                    or ``"per_class_f1"``. Values are assumed in [0, 1].
        title: Plot title string.
        filename: Output PNG filename.
        colorbar_label: Label for the colour bar axis.
    """
    import numpy as np

    # Collect ordered class names from the first run (same across all runs)
    first_metrics = runs[0]["metrics"][metric_key]
    class_names = list(first_metrics.keys())
    model_names = [r["config"]["model"] for r in runs]

    n_models = len(model_names)
    n_classes = len(class_names)

    matrix = np.zeros((n_models, n_classes))
    for i, run in enumerate(runs):
        per_class = run["metrics"][metric_key]
        for j, cls in enumerate(class_names):
            matrix[i, j] = per_class.get(cls, 0.0) * 100

    fig, ax = plt.subplots(figsize=(max(10, n_classes * 1.1), max(4, n_models * 0.9)))

    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, label=colorbar_label)

    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_models))
    ax.set_xticklabels(class_names, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(model_names, fontsize=9)

    midpoint = (matrix.max() + matrix.min()) / 2
    for i in range(n_models):
        for j in range(n_classes):
            ax.text(
                j, i, f"{matrix[i, j]:.1f}",
                ha="center", va="center", fontsize=8,
                color="white" if matrix[i, j] > midpoint else "black",
            )

    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=150)
    plt.close()


def plot_training_efficiency(
    runs: list[dict],
    output_dir: Path,
    plt,
    *,
    filename: str = "sequential_training_efficiency.png",
) -> None:
    """Dual-axis chart: epochs trained (bars) and wall-clock time (line) per model.

    Args:
        runs: List of completed run result dicts.
        output_dir: Directory to save the plot.
        plt: The matplotlib.pyplot module (passed by caller).
        filename: Output PNG filename.
    """
    import numpy as np

    model_colors = _get_model_colors()
    model_names  = [r["config"]["model"] for r in runs]
    epochs_list  = [r["metrics"]["epochs_trained"] for r in runs]
    time_minutes = [r["metrics"]["total_training_time_seconds"] / 60 for r in runs]
    colors       = [model_colors.get(m, "#999999") for m in model_names]

    x = np.arange(len(model_names))
    fig, ax1 = plt.subplots(figsize=(max(8, len(runs) * 1.6), 6))

    bars = ax1.bar(x, epochs_list, color=colors, alpha=0.8, label="Epochs trained")
    ax1.set_ylabel("Epochs trained")
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=15, ha="right")
    ax1.set_ylim(0, max(epochs_list) * 1.2)

    for bar, ep in zip(bars, epochs_list):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,
            str(ep),
            ha="center", va="center", fontsize=9, fontweight="bold", color="white",
        )

    ax2 = ax1.twinx()
    ax2.plot(x, time_minutes, color="#e74c3c", marker="o", linewidth=2,
             markersize=8, label="Training time (min)")
    ax2.set_ylabel("Training time (min)")
    ax2.set_ylim(0, max(time_minutes) * 1.25)

    for xi, tm in zip(x, time_minutes):
        ax2.text(
            xi,
            tm + max(time_minutes) * 0.03,
            f"{tm:.1f}m",
            ha="center", va="bottom", fontsize=9, color="#e74c3c",
        )

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")

    ax1.set_title("Training Efficiency: Epochs Run & Wall-Clock Time")
    ax1.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=150)
    plt.close()
