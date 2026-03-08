"""Training subpackage — configuration, training loops, and experiment runners."""

from .configs import ModelConfig, OptimizerFactory, SchedulerFactory
from .trainer import ModelTrainer, TrainingResults
from .sequential import run_sequential, SAMPLING_MAP
from .grid_search import GridSearch, GridSearchConfig, AblationConfig

__all__ = [
    "ModelConfig",
    "OptimizerFactory",
    "SchedulerFactory",
    "ModelTrainer",
    "TrainingResults",
    "run_sequential",
    "SAMPLING_MAP",
    "GridSearch",
    "GridSearchConfig",
    "AblationConfig",
]
