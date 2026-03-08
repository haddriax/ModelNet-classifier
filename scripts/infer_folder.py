"""Batch folder inference — run a trained checkpoint over every .off file in a folder.

Workflow:

1. **Folder picker** (GUI) — select any directory containing ``.off`` files.
2. **Checkpoint picker** (GUI) — select a ``.pth`` file from ``models/``.
3. **Inference** is run on every ``.off`` file found recursively in the folder.
4. A result row is printed for each file::

       filename | true class | predicted class | confidence | ✓/✗

   The true class is extracted from the standard ModelNet filename convention
   ``class_NNN.off`` (e.g. ``night_stand_0042.off`` → ``night_stand``).
5. A summary line reports the overall accuracy over all labelled files.

Usage::

    python -m scripts.infer_folder
"""

import re
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import numpy as np
import torch

from src.builders.mesh_3D_builder import Mesh3DBuilder
from src.config import DATA_DIR, MODELS_DIR
from src.deep_learning.inference import (
    detect_dataset_from_path,
    load_model_from_checkpoint,
    parse_checkpoint_config,
    run_inference,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_TRUE_CLASS_RE = re.compile(r'^(.+)_\d+\.off$')


def _pick_folder() -> Path | None:
    """Open a directory-picker dialog rooted at DATA_DIR."""
    initial = DATA_DIR if DATA_DIR.exists() else Path.cwd()
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askdirectory(
        title="Select folder containing .off files",
        initialdir=str(initial),
    )
    root.destroy()
    return Path(path) if path else None


def _pick_checkpoint() -> Path | None:
    """Open a file-picker dialog for a .pth checkpoint rooted at MODELS_DIR."""
    initial = MODELS_DIR if MODELS_DIR.exists() else Path.cwd()
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Select model checkpoint (.pth)",
        initialdir=str(initial),
        filetypes=[("PyTorch checkpoint", "*.pth"), ("All files", "*.*")],
    )
    root.destroy()
    return Path(path) if path else None


def _extract_true_class(filename: str) -> str | None:
    """Return the class name encoded in ``class_NNN.off``, or *None*.

    Works correctly for multi-word class names such as ``night_stand_0001.off``.
    """
    m = _TRUE_CLASS_RE.match(filename)
    return m.group(1) if m else None


def _build_class_map(data_dir: Path) -> dict[int, str]:
    """Build an ``{index: class_name}`` map from sorted subdirectories of *data_dir*."""
    classes = sorted(d.name for d in data_dir.iterdir() if d.is_dir())
    return {i: name for i, name in enumerate(classes)}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # ── 1. Pick folder ───────────────────────────────────────────────────────
    folder = _pick_folder()
    if not folder:
        print("No folder selected — exiting.")
        return

    off_files = sorted(folder.rglob("*.off"))
    if not off_files:
        print(f"No .off files found in: {folder}")
        return

    print(f"\nFound {len(off_files)} .off file(s) in: {folder}")

    # ── 2. Pick and load checkpoint ──────────────────────────────────────────
    ckpt_path = _pick_checkpoint()
    if not ckpt_path:
        print("No checkpoint selected — exiting.")
        return

    dataset_info = detect_dataset_from_path(ckpt_path)
    if dataset_info is None:
        print(
            "ERROR: Could not detect dataset from checkpoint path.\n"
            "       Ensure the checkpoint lives under a folder named "
            "'modelnet10' or 'modelnet40'."
        )
        return
    data_dir, num_classes = dataset_info

    ckpt_config = parse_checkpoint_config(ckpt_path)
    if ckpt_config is None:
        print(
            "ERROR: Checkpoint filename does not match the naming convention.\n"
            f"       Expected : {{ModelName}}_{{sampling}}_pts{{N}}_bs{{B}}[_best].pth\n"
            f"       Got      : {ckpt_path.name}"
        )
        return
    model_class, n_points, sampling = ckpt_config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model_from_checkpoint(ckpt_path, model_class, num_classes, device)
    class_map = _build_class_map(data_dir)

    # ── 3. Print run header ──────────────────────────────────────────────────
    print()
    print(f"  Model     : {model_class.__name__}  ({num_classes} classes)")
    print(f"  Checkpoint: {ckpt_path.name}")
    print(f"  Sampling  : {sampling.name.lower()}  |  Points: {n_points}")
    print(f"  Device    : {device}")
    print()

    W_FILE = 36
    W_TRUE = 20
    W_PRED = 20
    W_CONF =  7
    SEP = "-" * (W_FILE + W_TRUE + W_PRED + W_CONF + 4)

    print(f"{'File':<{W_FILE}}{'True class':<{W_TRUE}}{'Predicted':<{W_PRED}}{'Conf':>{W_CONF}}")
    print(SEP)

    # ── 4. Inference loop ────────────────────────────────────────────────────
    correct        = 0
    unknown        = 0
    fallback_count = 0
    error_count    = 0

    for off_file in off_files:
        true_class = _extract_true_class(off_file.name)

        # -- load mesh --
        try:
            mesh = Mesh3DBuilder.from_off_file(off_file)
        except Exception as exc:
            print(f"  [ERROR] {off_file.name} (load): {str(exc).splitlines()[0]}")
            error_count += 1
            continue

        # -- sample points (with vertex fallback for degenerate meshes) --
        used_fallback = False
        try:
            points = mesh.sample_points(n_points=n_points, method=sampling)
        except Exception as exc:
            first_line = str(exc).splitlines()[0]
            print(f"  [WARN]  {off_file.name}: sampling failed — using vertex fallback"
                  f" ({first_line})")
            idx    = np.random.choice(len(mesh.vertices), n_points,
                                      replace=len(mesh.vertices) < n_points)
            points = mesh.vertices[idx].astype(np.float32)
            used_fallback = True
            fallback_count += 1

        # -- inference --
        try:
            pred_idx, conf = run_inference(model, points, device)
        except Exception as exc:
            print(f"  [ERROR] {off_file.name} (infer): {str(exc).splitlines()[0]}")
            error_count += 1
            continue

        pred_class   = class_map.get(pred_idx, f"idx_{pred_idx}")
        true_display = true_class if true_class is not None else "?"

        if true_class is None:
            status   = "?†" if used_fallback else "?"
            unknown += 1
        elif true_class == pred_class:
            status   = "✓†" if used_fallback else "✓"
            correct += 1
        else:
            status = "✗†" if used_fallback else "✗"

        print(
            f"{off_file.name:<{W_FILE}}"
            f"{true_display:<{W_TRUE}}"
            f"{pred_class:<{W_PRED}}"
            f"{conf * 100:{W_CONF - 1}.1f}%"
            f"  {status}"
        )

    # ── 5. Summary ───────────────────────────────────────────────────────────
    print(SEP)

    total_labelled = len(off_files) - unknown - error_count
    acc = correct / total_labelled * 100 if total_labelled > 0 else 0.0

    if unknown:
        print(f"  ({unknown} file(s) had no recognisable class label in their name)")
    if fallback_count:
        print(f"  ({fallback_count} file(s) used vertex-sampling fallback  †)")
    if error_count:
        print(f"  ({error_count} file(s) skipped due to unrecoverable errors)")
    print(f"\n  Results: {correct}/{total_labelled} correct  —  accuracy {acc:.1f}%\n")


if __name__ == "__main__":
    main()
