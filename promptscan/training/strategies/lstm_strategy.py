#!/usr/bin/env python3
"""
LSTM training strategy.
"""

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from ...config import ModelConfig
from ...models.lstm_model import LSTMModel
from ...processors.word_processor import WordProcessor
from ..base_trainer import BaseTrainer, TrainingStrategy


class LSTMTrainer(BaseTrainer):
    """LSTM-specific trainer."""

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create AdamW optimizer for LSTM."""
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=1e-4,
        )

    def _prepare_batch(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Prepare batch for LSTM forward pass."""
        inputs = {"input_ids": batch["input_ids"].to(self.device)}
        labels = batch["label"].to(self.device)
        return inputs, labels

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with memory management."""
        # Clear GPU cache at start of epoch
        if self.device == "cuda":
            torch.cuda.empty_cache()

        metrics = super().train_epoch()

        # Clear GPU cache at end of epoch
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return metrics


class LSTMTrainingStrategy(TrainingStrategy):
    """Training strategy for LSTM models."""

    def create_model(self, config: ModelConfig) -> nn.Module:
        """Create LSTM model."""
        # Estimate vocab size (will be updated after processor is created)
        vocab_size = 10000  # Default, will be updated

        return LSTMModel(
            vocab_size=vocab_size,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            num_classes=2,
        )

    def create_processor(self, config: ModelConfig) -> Any:
        """Create word processor for LSTM."""
        return WordProcessor(
            max_length=100,  # LSTM-specific max length
            min_freq=2,
        )

    def create_dataset(self, data: Dict, processor: Any):
        """Create dataset for LSTM training."""
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
        """Create LSTM trainer."""
        # Update model vocab size based on processor
        if hasattr(processor, "vocab"):
            model.embedding = nn.Embedding(
                len(processor.vocab),
                config.embedding_dim,
                padding_idx=0,
            )

        return LSTMTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            processor=processor,
        )

    def get_collate_fn(self):
        """LSTM doesn't need special collate function."""
        return None
