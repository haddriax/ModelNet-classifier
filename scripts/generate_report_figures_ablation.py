"""Generate report figures from an ablation (grid search) JSON result.

Opens a file-picker dialog rooted at ``results/grid/``, loads the selected
``ablation_results.json`` (or ``ablation_results_intermediate.json``), and
generates the four figures used in the LaTeX report:

* ``sampling_comparison.png``  — Figure 6  (sampling method comparison)
* ``npoints_effect.png``       — Figure 7  (effect of number of points)
* ``batchsize_effect.png``     — Figure 8  (effect of batch size)
* ``model_heatmap.png``        — Figure 9  (model × config heatmap)

All figures are saved next to the selected JSON file.

Note: ``accuracy_comparison.png`` and ``ablation_training_efficiency.png`` are
also generated as side-effects of the underlying plotting pipeline; they are not
used in the report and can be ignored.

Run the grid search first if no ablation JSON exists for ModelNet10::

    python -m scripts.grid_training --dataset modelnet10

Usage::

    python -m scripts.generate_report_figures_ablation
"""

import tkinter as tk
from tkinter import filedialog
from pathlib import Path

from src.config import RESULTS_DIR
from src.deep_learning.plotting import create_ablation_plots

# ── Constants ──────────────────────────────────────────────────────────────────

_INITIAL_DIR = RESULTS_DIR / "grid"

# Figures that are actually used in the LaTeX report (for the summary printout)
_REPORT_FIGURES = [
    "sampling_comparison.png",
    "npoints_effect.png",
    "batchsize_effect.png",
    "model_heatmap.png",
]


# ── GUI helper ─────────────────────────────────────────────────────────────────

def _pick_json() -> Path | None:
    """Open a file-picker dialog and return the selected JSON path, or None."""
    initial = _INITIAL_DIR if _INITIAL_DIR.exists() else Path.cwd()

    root = tk.Tk()
    root.withdraw()
    raw = filedialog.askopenfilename(
        title="Select an ablation_results.json file",
        initialdir=str(initial),
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
    )
    root.destroy()

    return Path(raw) if raw else None


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    json_path = _pick_json()
    if json_path is None:
        print("No file selected. Exiting.")
        return

    if not json_path.exists():
        print(f"File not found: {json_path}")
        return

    output_dir = json_path.parent
    print(f"\nGenerating report figures from:\n  {json_path}")
    print(f"Output directory:\n  {output_dir}\n")

    create_ablation_plots(json_path, output_dir)

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n── Report figures (" + "─" * 45)
    for name in _REPORT_FIGURES:
        dest = output_dir / name
        status = "✓" if dest.exists() else "✗ MISSING"
        print(f"  [{status}] {dest}")

    print("\n── To insert in the LaTeX report, copy to figures/ (keep original names):")
    for name in _REPORT_FIGURES:
        src = output_dir / name
        if src.exists():
            print(f"  {name}  →  figures/{name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
