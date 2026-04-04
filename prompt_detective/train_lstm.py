#!/usr/bin/env python3
"""
Training script for LSTM model.
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .models.lstm_model import LSTMModel
from .processors.word_processor import WordProcessor
from .training.data_loader import load_data_from_parquet
from .utils.data_loader import PromptDataset
from .utils.device import get_device


class LSTMTrainer:
    """Trainer for LSTM model."""

    def __init__(self, model, train_loader, val_loader, processor, device="cpu"):
        self.device = get_device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.processor = processor

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-3)

        # Track best metrics
        self.best_train_acc = 0.0
        self.best_val_acc = 0.0

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        # Add progress bar
        pbar = tqdm(self.train_loader, desc="Training", leave=False)

        for batch in pbar:
            inputs = batch["input_ids"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model({"input_ids": inputs})
            loss = self.criterion(outputs, labels)
            loss.backward()

            # Gradient clipping to prevent instability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar
            pbar.set_postfix(
                {"loss": f"{loss.item():.4f}", "acc": f"{correct / total:.2%}"}
            )

            # Clear unnecessary tensors to free memory
            del inputs, labels, outputs, loss

        return total_loss / len(self.train_loader), correct / total

    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", leave=False):
                inputs = batch["input_ids"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model({"input_ids": inputs})
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Clear unnecessary tensors to free memory
                del inputs, labels, outputs, loss

        return total_loss / len(self.val_loader), correct / total

    def train(self, epochs=20, patience=3):
        """Train the model with early stopping."""
        best_val_acc = 0
        patience_counter = 0

        try:
            for epoch in range(epochs):
                print(f"\nEpoch {epoch + 1}/{epochs}")

                train_loss, train_acc = self.train_epoch()
                val_loss, val_acc = self.validate()

                # Update best metrics
                self.best_train_acc = max(self.best_train_acc, train_acc)
                self.best_val_acc = max(self.best_val_acc, val_acc)

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

                # Clear GPU cache between epochs
                if self.device == "cuda":
                    torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print("\n" + "=" * 60)
                print("❌ GPU OUT OF MEMORY ERROR!")
                print("=" * 60)
                if self.device == "cuda":
                    try:
                        allocated = torch.cuda.memory_allocated(0) / 1024**3
                        reserved = torch.cuda.memory_reserved(0) / 1024**3
                        print(f"GPU Memory Allocated: {allocated:.2f} GB")
                        print(f"GPU Memory Reserved: {reserved:.2f} GB")
                    except:
                        pass
                print("\nSuggestions to fix this:")
                print(
                    "  1. Reduce batch size (current training uses batch size from args)"
                )
                print("  2. Reduce model size (hidden_dim, embedding_dim)")
                print("  3. Use CPU instead: --device cpu")
                print("  4. Close other GPU applications")
                print(
                    "\nExample: python -m prompt_detective.train_lstm --batch-size 8 --device auto"
                )
                print("=" * 60)
                raise
            else:
                raise
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user")
            print(f"Best validation accuracy so far: {best_val_acc:.2%}")
            raise


def train_lstm_model(
    train_path="data/train.parquet",
    val_path="data/val.parquet",
    test_path="data/test.parquet",
    model_path="models/lstm_best.pt",
    epochs=20,
    batch_size=16,
    learning_rate=1e-3,
    device="auto",
):
    """Train LSTM model."""
    print("=" * 60)
    print("LSTM Model Training")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    train_data, val_data, test_data = load_data_from_parquet(
        train_path, val_path, test_path
    )
    print(f"  Train samples: {len(train_data)}")
    print(f"  Val samples: {len(val_data)}")
    print(f"  Test samples: {len(test_data)}")

    # Create processor and build vocabulary
    print("\nBuilding vocabulary...")
    processor = WordProcessor(max_length=100)
    processor.build_vocab([item["text"] for item in train_data])
    print(f"  Vocabulary size: {len(processor.vocab)}")

    # Create datasets
    train_dataset = PromptDataset(train_data, processor)
    val_dataset = PromptDataset(val_data, processor)

    # Determine optimal num_workers
    device_str = get_device(device)
    num_workers = 2 if device_str == "cuda" else 0  # Use workers only for GPU
    pin_memory = device_str == "cuda"  # Pin memory for faster GPU transfer

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Create model
    print("\nInitializing model...")
    model = LSTMModel(vocab_size=len(processor.vocab))
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Update optimizer with custom learning rate
    trainer = LSTMTrainer(model, train_loader, val_loader, processor, device=device)
    trainer.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    print("\nTraining configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Max epochs: {epochs}")
    print(f"  Device: {trainer.device}")

    # Show memory info for CUDA
    if trainer.device == "cuda":
        try:
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  GPU Memory Available: {total_mem:.2f} GB")
            print(f"  GPU Memory Currently Used: {allocated:.2f} GB")
        except:
            pass

    # Train
    print("\n" + "=" * 60)
    trainer.train(epochs=epochs)
    print("=" * 60)

    # Save model with tracked metrics
    print(f"\nSaving model to {model_path}...")
    model.save(
        model_path,
        processor,
        train_acc=trainer.best_train_acc,
        val_acc=trainer.best_val_acc,
    )

    print("\n✓ Training complete!")
    print(f"  Best train accuracy: {trainer.best_train_acc:.2%}")
    print(f"  Best val accuracy: {trainer.best_val_acc:.2%}")
    print(f"  Model saved to: {model_path}")

    return model, processor


if __name__ == "__main__":
    train_lstm_model()
