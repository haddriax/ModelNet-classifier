"""Demo entry point — quick-start for new users.

Runs the full ModelNet10 pipeline in three steps:

1. **Dataset setup** — checks and fixes the ModelNet10 directory layout
   (idempotent; safe to run even when the data is already correctly placed).

2. **TensorBoard** — launches ``tensorboard --logdir=runs`` as a background
   process on port 6006.  Open http://localhost:6006 to monitor training live.

3. **Quick sequential training** — trains all five model architectures on
   ModelNet10 for 10 epochs each (512 points, batch size 32).  Results,
   metrics, and plots are saved automatically to a timestamped directory under
   ``results/sequential/modelnet10/``.

TensorBoard keeps running after training completes so you can explore the
loss and accuracy curves at your leisure.

Usage::

    python -m scripts.main
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

from scripts.setup_dataset import fix_dataset
from src.config import DATA_DIR, MODELS_DIR, RESULTS_DIR
from src.deep_learning.training import ModelConfig, run_sequential

# ---------------------------------------------------------------------------
# Demo settings — deliberately lightweight for a quick first run
# ---------------------------------------------------------------------------

_N_POINTS   = 512   # smaller than the full 1024 for speed
_BATCH_SIZE = 32
_EPOCHS     = 10    # just enough to see the learning curves take shape

_DEMO_CONFIGS: dict[str, ModelConfig] = {
    "PointNet":         ModelConfig(sampling="uniform"),
    "SimplePointNet":   ModelConfig(sampling="uniform"),
    "DGCNN":            ModelConfig(sampling="uniform"),
    "PointNetPP":       ModelConfig(sampling="fps"),
    "PointTransformer": ModelConfig(sampling="fps"),
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  ModelNet10 Classifier — Demo Mode")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1/3  Dataset layout check
    # ------------------------------------------------------------------
    print("\n[1/3] Checking ModelNet10 dataset layout …")
    fix_dataset(DATA_DIR.parent)  # data/ModelNet10/

    # ------------------------------------------------------------------
    # 2/3  Launch TensorBoard in the background
    # ------------------------------------------------------------------
    print("\n[2/3] Starting TensorBoard …")
    Path("runs").mkdir(exist_ok=True)
    tb_proc = None
    try:
        tb_proc = subprocess.Popen(
            [sys.executable, "-m", "tensorboard", "--logdir=runs", "--port=6006"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("  TensorBoard running → http://localhost:6006")
    except Exception as exc:
        print(f"  [warn] Could not start TensorBoard: {exc}")
        print("         You can start it manually: tensorboard --logdir=runs")

    # ------------------------------------------------------------------
    # 3/3  Quick sequential training
    # ------------------------------------------------------------------
    n_models = len(_DEMO_CONFIGS)
    print(f"\n[3/3] Training {n_models} models × {_EPOCHS} epochs "
          f"({_N_POINTS} pts, batch {_BATCH_SIZE}) …\n")

    timestamp   = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    results_dir = RESULTS_DIR / "sequential" / "modelnet10" / timestamp
    models_dir  = MODELS_DIR  / "sequential" / "modelnet10" / timestamp

    run_sequential(
        _DEMO_CONFIGS,
        n_points=_N_POINTS,
        batch_size=_BATCH_SIZE,
        epochs=_EPOCHS,
        data_dir=DATA_DIR,
        results_dir=results_dir,
        models_dir=models_dir,
    )

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("  Demo complete!")
    print(f"  Results → {results_dir}")
    if tb_proc is not None:
        print("  TensorBoard → http://localhost:6006  (still running)")
        print("  Stop it with Ctrl+C, or: kill the tensorboard process")
    print("=" * 60)
