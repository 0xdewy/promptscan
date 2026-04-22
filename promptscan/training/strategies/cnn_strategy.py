#!/usr/bin/env python3
"""
CNN training strategy.
"""

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from ...config import ModelConfig
from ...models.cnn_model import SimpleCNN
from ...processors.word_processor import WordProcessor
from ..base_trainer import BaseTrainer, TrainingStrategy


class CNNTrainer(BaseTrainer):
    """CNN-specific trainer."""

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create AdamW optimizer for CNN."""
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=1e-4,
        )

    def _create_scheduler(self):
        """Create ReduceLROnPlateau scheduler for CNN."""
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        return ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        )

    def _prepare_batch(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Prepare batch for CNN forward pass."""
        inputs = {"input_ids": batch["input_ids"].to(self.device)}
        labels = batch["label"].to(self.device)
        return inputs, labels


class CNNTrainingStrategy(TrainingStrategy):
    """Training strategy for CNN models."""

    def create_model(self, config: ModelConfig) -> nn.Module:
        """Create CNN model."""
        # Estimate vocab size (will be updated after processor is created)
        vocab_size = 10000  # Default, will be updated

        return SimpleCNN(
            vocab_size=vocab_size,
            embedding_dim=config.embedding_dim,
            num_filters=config.num_filters,
            num_classes=2,
        )

    def create_processor(self, config: ModelConfig) -> Any:
        """Create word processor for CNN."""
        return WordProcessor(
            max_length=config.max_length,
            min_freq=2,
        )

    def create_dataset(self, data: Dict, processor: Any):
        """Create dataset for CNN training."""
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
        resume: bool = False,
    ) -> BaseTrainer:
        """Create CNN trainer."""
        if hasattr(processor, "vocab") and hasattr(model, "embedding"):
            old_vocab_size = model.embedding.num_embeddings

            max_token_id = max((v for v in processor.vocab.values() if isinstance(v, int)), default=0)
            required_size = max_token_id + 1

            if required_size > old_vocab_size:
                embedding_dim = model.embedding.embedding_dim
                new_embedding = nn.Embedding(
                    required_size,
                    embedding_dim,
                    padding_idx=0,
                )

                copy_size = min(old_vocab_size, required_size)
                with torch.no_grad():
                    new_embedding.weight[:copy_size] = model.embedding.weight[
                        :copy_size
                    ]

                if required_size > old_vocab_size:
                    nn.init.uniform_(new_embedding.weight[old_vocab_size:], -0.1, 0.1)

                model.embedding = new_embedding
                print(f"  Resized CNN embedding: {old_vocab_size} -> {required_size}")

        return CNNTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            processor=processor,
        )

    def get_collate_fn(self):
        """Collate function for CNN that stacks individual tensors."""

        def collate_fn(batch):
            # Stack input_ids
            input_ids = torch.stack([item["input_ids"] for item in batch])

            # Stack labels
            labels = torch.stack([item["label"] for item in batch])

            # Create result dictionary
            result = {
                "input_ids": input_ids,
                "label": labels,
            }

            # Add any additional fields from batch items
            for key in batch[0].keys():
                if key not in ["input_ids", "label"]:
                    result[key] = torch.stack([item[key] for item in batch])

            return result

        return collate_fn
