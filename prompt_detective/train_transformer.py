#!/usr/bin/env python3
"""
Training script for Transformer (DistilBERT) model.
"""

from typing import Dict, List

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

from .models.transformer_model import TransformerModel
from .processors.subword_processor import SubwordProcessor
from .utils.device import get_device


class TransformerDataset(Dataset):
    """Dataset for transformer models."""

    def __init__(self, data: List[Dict], processor: SubwordProcessor):
        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.processor.encode(item["text"])

        # Convert to tensors
        result = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(item["label"], dtype=torch.long),
        }
        return result


class TransformerTrainer:
    """Trainer for transformer model."""

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device="cpu",
        learning_rate=2e-5,
        num_warmup_steps=0,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

        # Learning rate scheduler
        num_training_steps = len(train_loader) * 20  # 20 epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in self.train_loader:
            inputs = {
                "input_ids": batch["input_ids"].to(self.device),
                "attention_mask": batch["attention_mask"].to(self.device),
            }
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        return total_loss / len(self.train_loader), correct / total

    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in self.val_loader:
                inputs = {
                    "input_ids": batch["input_ids"].to(self.device),
                    "attention_mask": batch["attention_mask"].to(self.device),
                }
                labels = batch["label"].to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return total_loss / len(self.val_loader), correct / total

    def train(self, epochs=3, patience=2):
        """Train the model with early stopping."""
        best_val_acc = 0
        patience_counter = 0

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            print(f"Epoch {epoch + 1}/{epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                print(f"  ✓ New best model (val acc: {val_acc:.2%})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break


def load_data_from_parquet(train_path, val_path, test_path):
    """Load training data from parquet files."""
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    # Convert to list of dictionaries
    train_data = []
    for _, row in train_df.iterrows():
        train_data.append(
            {"text": row["text"], "label": 1 if row["is_injection"] else 0}
        )

    val_data = []
    for _, row in val_df.iterrows():
        val_data.append({"text": row["text"], "label": 1 if row["is_injection"] else 0})

    test_data = []
    for _, row in test_df.iterrows():
        test_data.append(
            {"text": row["text"], "label": 1 if row["is_injection"] else 0}
        )

    return train_data, val_data, test_data


def collate_fn(batch):
    """Collate function for transformer dataset."""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])

    return {"input_ids": input_ids, "attention_mask": attention_mask, "label": labels}


def train_transformer_model(
    train_path="data/train.parquet",
    val_path="data/val.parquet",
    test_path="data/test.parquet",
    model_path="models/transformer_best.pt",
    epochs=3,  # Fewer epochs for transformer (fine-tuning)
    batch_size=16,  # Smaller batch size for transformer
    learning_rate=2e-5,  # Standard for fine-tuning
    device="cpu",
):
    """Train transformer model."""
    from .config import ModelConfig

    # Create configuration
    config = ModelConfig(
        model_type="transformer",
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        transformer_model="distilbert-base-uncased",
        max_length=128,
    )

    # Validate configuration
    errors = config.validate()
    if errors:
        raise ValueError(f"Invalid configuration: {errors}")

    # Convert device string to actual device
    device = get_device(config.device)

    # Load data
    train_data, val_data, test_data = load_data_from_parquet(
        train_path, val_path, test_path
    )

    # Create processor
    processor = SubwordProcessor(
        model_name=config.transformer_model, max_length=config.max_length
    )

    # Create datasets
    train_dataset = TransformerDataset(train_data, processor)
    val_dataset = TransformerDataset(val_data, processor)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Create model
    model = TransformerModel(model_name=config.transformer_model, num_classes=2)

    # Train
    trainer = TransformerTrainer(
        model,
        train_loader,
        val_loader,
        device=device,
        learning_rate=config.learning_rate,
    )
    trainer.train(epochs=config.epochs)

    # Save model
    model.save(
        model_path,
        processor,
        train_acc=trainer.train_epoch()[1],
        val_acc=trainer.validate()[1],
    )

    print(f"Transformer model saved to {model_path}")
    print("Model: distilbert-base-uncased")

    return model, processor


if __name__ == "__main__":
    train_transformer_model()
