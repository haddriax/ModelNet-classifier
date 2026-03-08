"""Generate report figures from a sequential training JSON result.

Opens a file-picker dialog rooted at ``results/sequential/``, loads the
selected ``sequential_results.json``, and generates the three figures used in
the LaTeX report:

* ``sequential_model_comparison.png``   — Figure 10 (model comparison bar chart)
* ``sequential_per_class_f1.png``       — Figure 11 (per-class F1 heatmap)
* ``sequential_training_efficiency.png``— Figure 12 (accuracy vs. training time)

All figures are saved next to the selected JSON file.

Note: ``sequential_per_class_accuracy.png`` is also generated as a side-effect
of the underlying plotting pipeline; it is not used in the report and can be
ignored.

Usage::

    python -m scripts.generate_report_figures_sequential
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

from src.config import RESULTS_DIR
from src.deep_learning.plotting import plot_sequential_results

# ── Constants ──────────────────────────────────────────────────────────────────

_INITIAL_DIR = RESULTS_DIR / "sequential"

# Figures that are actually used in the LaTeX report (for the summary printout)
_REPORT_FIGURES = [
    "sequential_model_comparison.png",
    "sequential_per_class_f1.png",
    "sequential_training_efficiency.png",
]


# ── GUI helper ─────────────────────────────────────────────────────────────────

def _pick_json() -> Path | None:
    """Open a file-picker dialog and return the selected JSON path, or None."""
    initial = _INITIAL_DIR if _INITIAL_DIR.exists() else Path.cwd()

    root = tk.Tk()
    root.withdraw()
    raw = filedialog.askopenfilename(
        title="Select a sequential_results.json file",
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

    plot_sequential_results(json_path, output_dir)

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n── Report figures (" + "─" * 45)
    for name in _REPORT_FIGURES:
        dest = output_dir / name
        status = "✓" if dest.exists() else "✗ MISSING"
        print(f"  [{status}] {dest}")

    print("\n── To insert in the LaTeX report, copy to figures/:")
    rename_map = {
        "sequential_model_comparison.png": "mn10_model_comparison.png",
        "sequential_per_class_f1.png": "mn10_per_class_f1.png",
        "sequential_training_efficiency.png": "mn10_training_efficiency.png",
    }
    for src_name, dst_name in rename_map.items():
        src = output_dir / src_name
        if src.exists():
            print(f"  {src_name}  →  figures/{dst_name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
