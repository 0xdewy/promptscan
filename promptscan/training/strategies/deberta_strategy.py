#!/usr/bin/env python3
"""
DeBERTa training strategy.
"""

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup

from ...config import ModelConfig
from ...models.deberta_model import DeBERTaModel
from ...processors.subword_processor import SubwordProcessor
from ..base_trainer import BaseTrainer, TrainingStrategy


class DeBERTaTrainer(BaseTrainer):
    """DeBERTa-specific trainer."""

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create AdamW optimizer for DeBERTa."""
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01,
        )

    def _create_scheduler(self):
        """Create learning rate scheduler with warmup for DeBERTa."""
        num_training_steps = len(self.train_loader) * self.config.epochs
        num_warmup_steps = int(0.1 * num_training_steps)

        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

    def _prepare_batch(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Prepare batch for DeBERTa forward pass."""
        inputs = {}
        for key in batch.keys():
            if key != "label":
                inputs[key] = batch[key].to(self.device)

        labels = batch["label"].to(self.device)
        return inputs, labels

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        return super().train_epoch()


class DeBERTaTrainingStrategy(TrainingStrategy):
    """Training strategy for DeBERTa models."""

    def create_model(self, config: ModelConfig) -> nn.Module:
        """Create DeBERTa model."""
        model_name = getattr(config, "deberta_model", "microsoft/deberta-v3-small")
        return DeBERTaModel(
            model_name=model_name,
            num_classes=2,
        )

    def create_processor(self, config: ModelConfig) -> Any:
        """Create subword processor for DeBERTa."""
        model_name = getattr(config, "deberta_model", "microsoft/deberta-v3-small")
        return SubwordProcessor(
            model_name=model_name,
            max_length=config.max_length,
        )

    def create_dataset(self, data: Dict, processor: Any):
        """Create dataset for DeBERTa training."""
        from ...training.data_loader import TextDataset

        return TextDataset(data, processor)

    def create_trainer(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        config: ModelConfig,
        processor: Any,
        resume: bool = False,
    ) -> BaseTrainer:
        """Create DeBERTa trainer."""
        return DeBERTaTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            processor=processor,
        )

    def get_collate_fn(self):
        """DeBERTa needs special collate function for batching."""

        def collate_fn(batch):
            """Collate function for DeBERTa dataset."""
            keys = batch[0].keys()
            result = {}

            for key in keys:
                if key == "label":
                    result[key] = torch.stack([item[key] for item in batch])
                else:
                    result[key] = torch.stack([item[key] for item in batch])

            return result

        return collate_fn
