"""Deep learning subpackage for point cloud classification.

Subpackages
-----------
training/   Configuration dataclasses, training loop, sequential and grid-search runners.
models/     Model architectures (PointNet, DGCNN, PointNetPP, PointTransformer, SimplePointNet).
plotting/   Ablation and sequential training visualisation utilities.
"""
# Intentionally minimal: heavy submodules (training, dataset_factory) import
# the dataset layer at module init time, so they are not eagerly re-exported
# here. Import them directly:
#
#   from src.deep_learning.training import ModelConfig, run_sequential
#   from src.deep_learning.models import ALL_MODELS
#   from src.deep_learning.plotting import create_ablation_plots
