#!/usr/bin/env python3
"""
Transformer training strategy.
"""

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup

from ...config import ModelConfig
from ...models.transformer_model import TransformerModel
from ...processors.subword_processor import SubwordProcessor
from ..base_trainer import BaseTrainer, TrainingStrategy


class TransformerTrainer(BaseTrainer):
    """Transformer-specific trainer."""

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create AdamW optimizer for transformer."""
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01,
        )

    def _create_scheduler(self):
        """Create learning rate scheduler with warmup for transformer."""
        num_training_steps = len(self.train_loader) * self.config.epochs
        num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup

        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

    def _prepare_batch(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Prepare batch for transformer forward pass."""
        # Include all keys except 'label' as inputs to transformer
        inputs = {}
        for key in batch.keys():
            if key != "label":
                inputs[key] = batch[key].to(self.device)

        labels = batch["label"].to(self.device)
        return inputs, labels

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with gradient clipping."""
        metrics = super().train_epoch()

        # Transformer-specific: gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        return metrics


class TransformerTrainingStrategy(TrainingStrategy):
    """Training strategy for transformer models."""

    def create_model(self, config: ModelConfig) -> nn.Module:
        """Create transformer model."""
        return TransformerModel(
            model_name=config.transformer_model,
            num_classes=2,
        )

    def create_processor(self, config: ModelConfig) -> Any:
        """Create subword processor for transformer."""
        return SubwordProcessor(
            model_name=config.transformer_model,
            max_length=config.max_length,
        )

    def create_dataset(self, data: Dict, processor: Any):
        """Create dataset for transformer training."""
        # Uses generic TextDataset from data_loader
        from ...training.data_loader import TextDataset

        return TextDataset(data, processor)

    def create_trainer(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        config: ModelConfig,
        processor: Any,
    ) -> BaseTrainer:
        """Create transformer trainer."""
        return TransformerTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            processor=processor,
        )

    def get_collate_fn(self):
        """Transformer needs special collate function for batching."""

        def collate_fn(batch):
            """Collate function for transformer dataset."""
            # Get all keys from first item
            keys = batch[0].keys()
            result = {}

            for key in keys:
                if key == "label":
                    result[key] = torch.stack([item[key] for item in batch])
                else:
                    # Stack tensors for input_ids, attention_mask, token_type_ids, etc.
                    result[key] = torch.stack([item[key] for item in batch])

            return result

        return collate_fn
