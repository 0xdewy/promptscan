"""
Unified training pipeline for prompt injection detection.
"""

from .base_trainer import BaseTrainer, TrainingStrategy
from .data_loader import (
    TextDataset,
    create_dataloaders,
    get_data_stats,
    load_data_from_parquet,
    print_data_stats,
)
from .pipeline import create_default_config, get_training_strategy, train_model

__all__ = [
    "BaseTrainer",
    "TrainingStrategy",
    "load_data_from_parquet",
    "create_dataloaders",
    "TextDataset",
    "get_data_stats",
    "print_data_stats",
    "train_model",
    "get_training_strategy",
    "create_default_config",
]
