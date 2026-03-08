"""Sequential training entry point with curated per-model hyperparameters.

Trains each model with its own sampling method, learning rate, patience and
epoch budget rather than running a full Cartesian grid search.  One run per
model, executed sequentially.

Hyperparameters follow each model's original research paper as closely as
possible.  Paper-specific notes are included inline.

To use a custom scheduler or optimizer, pass a factory callable to ModelConfig::

    from torch.optim import AdamW
    from torch.optim.lr_scheduler import StepLR

    ModelConfig(
        sampling="fps",
        optimizer_factory=lambda params, lr: AdamW(params, lr=lr, weight_decay=1e-4),
        scheduler_factory=lambda opt, _: StepLR(opt, step_size=20, gamma=0.7),
    )

The implementation lives in :mod:`src.deep_learning.training.sequential`.

Usage::

    # ModelNet10 (default)
    python -m scripts.sequential_training --dataset modelnet10

    # ModelNet40
    python -m scripts.sequential_training --dataset modelnet40
"""

import torch
from torch.optim.lr_scheduler import ExponentialLR, StepLR

from src.deep_learning.training import ModelConfig, run_sequential

# ---------------------------------------------------------------------------
# Shared training settings (applies to all models unless overridden per-model)
# ---------------------------------------------------------------------------

N_POINTS   = 1024
BATCH_SIZE = 32

# ---------------------------------------------------------------------------
# Scheduler notes — ModelNet10 vs ModelNet40
# ---------------------------------------------------------------------------
# The original PointNet/PointNet++ papers decay LR ×0.7 every 200 000 gradient
# steps, calibrated for ModelNet40 (~9 843 steps/epoch → decay every ~20 epochs).
#
# On ModelNet10 (~3 991 samples, batch 32 → ~125 steps/epoch) the same formula
# gives a decay period of ~1 600 epochs — far beyond any budget used here.
# ExponentialLR with gamma=0.9827 therefore acts as a near-flat LR on MN10.
#
# Adaptation: StepLR decaying ×0.7 every 80 epochs, giving ~3 decay events over
# 250 epochs and a total LR reduction of ×0.7³ ≈ 0.343 — comparable to the
# paper's intent across training.  Switch to gamma=0.9956 / step_size=80 if you
# prefer to stay with ExponentialLR:
#   gamma = 0.7 ** (1/80) ≈ 0.9956
# ---------------------------------------------------------------------------

configs: dict[str, ModelConfig] = {
    # --------------------------------------------------------------
    # PointNet — Qi et al., CVPR 2017 (arXiv:1612.00593)
    # Paper config: Adam lr=0.001, exponential LR decay ×0.7 every
    # 200 K steps (≈ 20 epochs at batch 32 / MN40), 250 epochs,
    # uniform sampling, 1024 points.
    # ModelNet10 adaptation: StepLR ×0.7 every 80 epochs (~3 decays).
    # Reference: https://github.com/charlesq34/pointnet train.py
    # --------------------------------------------------------------
    "PointNet": ModelConfig(
        sampling="uniform",
        lr=0.001,
        epochs=50,
        patience=10,
        early_stop_metric="accuracy",
        scheduler_factory=lambda opt, _: StepLR(opt, step_size=80, gamma=0.9956),
    ),

    # --------------------------------------------------------------
    # PointNet++ — Qi et al., NeurIPS 2017 (arXiv:1706.02413)
    # Paper config: Adam lr=0.001, exponential decay ×0.7 every
    # 200 k steps, 250 epochs, batch 32, 1024 points, FPS sampling.
    # The hierarchical SA layers are designed around FPS; using
    # uniform sampling will still work but is sub-optimal.
    # ModelNet10 adaptation: same StepLR schedule as PointNet.
    # Increased patience: SA layers take longer to converge than the
    # flat PointNet MLP stack.
    # Reference: https://github.com/charlesq34/pointnet2 train.py
    # --------------------------------------------------------------
    "PointNetPP": ModelConfig(
        sampling="fps",
        lr=0.001,
        epochs=50,
        patience=10,
        early_stop_metric="accuracy",
        optimizer_factory=lambda params, lr: torch.optim.Adam(
            params, lr=lr, weight_decay=1e-4
        ),
        scheduler_factory=lambda opt, _: StepLR(opt, step_size=80, gamma=0.9956),
    ),

    # --------------------------------------------------------------
    # Point Transformer — Zhao et al., ICCV 2021 (arXiv:2012.09164)
    # Paper config: AdamW, lr=0.001, cosine annealing, weight
    # decay=0.05, 200 epochs, batch 32, 1024 points.
    # AdamW + cosine annealing is standard for transformer models.
    # Cosine annealing is a full-cycle schedule; stopping mid-cycle
    # undermines its convergence guarantee.
    # Reference: https://arxiv.org/abs/2012.09164
    # --------------------------------------------------------------
    "PointTransformer": ModelConfig(
        sampling="fps",
        lr=0.001,
        epochs=50,
        patience=10,
        early_stop_metric="accuracy",
        optimizer_factory=lambda params, lr: torch.optim.AdamW(
            params, lr=lr, weight_decay=0.05
        ),
        scheduler_factory=lambda opt, epochs: torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs, eta_min=1e-6
        ),
    ),

    # --------------------------------------------------------------
    # DGCNN — Wang et al., TOG 2019 (arXiv:1801.07829)
    # Paper config: Adam lr=0.001, step decay ×0.5 every 20 epochs,
    # 200 epochs, batch 32, 1024 uniform-sampled points.
    # Patience set to 2× step_size so a decay event always gets a
    # full recovery window before early stopping triggers.
    # Reference: https://github.com/WangYueFt/dgcnn train.py
    # --------------------------------------------------------------
    "DGCNN": ModelConfig(
        sampling="uniform",
        lr=0.001,
        epochs=50,
        patience=10,
        early_stop_metric="accuracy",
        scheduler_factory=lambda opt, _: StepLR(opt, step_size=20, gamma=0.5),
    ),
}


if __name__ == "__main__":
    import argparse
    from datetime import datetime
    from src.config import DATA_DIR, MODELNET40_DIR, MODELS_DIR, RESULTS_DIR

    # @todo: move deterministic configs into config files and ensure it's used in every training
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser(
        description="Sequential model training on ModelNet10 or ModelNet40."
    )
    parser.add_argument(
        "--dataset",
        choices=["modelnet10", "modelnet40"],
        default="modelnet40",
        help="Dataset to train on (default: modelnet10).",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    if args.dataset == "modelnet40":
        data_dir    = MODELNET40_DIR
        results_dir = RESULTS_DIR / "sequential" / "modelnet40" / timestamp
        models_dir  = MODELS_DIR  / "sequential" / "modelnet40" / timestamp
    else:
        data_dir    = DATA_DIR
        results_dir = RESULTS_DIR / "sequential" / "modelnet10" / timestamp
        models_dir  = MODELS_DIR  / "sequential" / "modelnet10" / timestamp

    run_sequential(
        configs,
        n_points=N_POINTS,
        batch_size=BATCH_SIZE,
        epochs=100,
        early_stop_metric="accuracy",
        data_dir=data_dir,
        results_dir=results_dir,
        models_dir=models_dir,
    )