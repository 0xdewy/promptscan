#!/usr/bin/env python3
"""
Base trainer class for unified training pipeline.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import ModelConfig
from ..utils.device import get_device


class BaseTrainer(ABC):
    """Abstract base class for all model trainers."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        config: ModelConfig,
        processor: Any = None,
    ):
        """
        Initialize trainer.

        Args:
            model: PyTorch model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            config: Training configuration
            processor: Text processor (optional)
        """
        self.config = config
        self.device = get_device(config.device)

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.processor = processor

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Learning rate scheduler (optional)
        self.scheduler = self._create_scheduler()

        # Performance optimizations
        self.use_amp = getattr(config, "use_amp", False) and self.device == "cuda"
        self.grad_accumulation_steps = getattr(config, "grad_accumulation_steps", 1)

        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        # Training state
        self.epoch = 0
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.history: List[Dict[str, float]] = []

    @abstractmethod
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer for the model."""
        pass

    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler (optional)."""
        return None

    @abstractmethod
    def _prepare_batch(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Prepare batch for model forward pass.

        Args:
            batch: Batch from DataLoader

        Returns:
            Tuple of (model_inputs, labels)
        """
        pass

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch with optional mixed precision and gradient accumulation.

        Returns:
            Dictionary with epoch metrics
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        # Reset gradient accumulation
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            # Prepare batch
            inputs, labels = self._prepare_batch(batch)

            # Mixed precision training
            if self.use_amp and self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                    # Scale loss for gradient accumulation
                    loss = loss / self.grad_accumulation_steps

                    # Backward pass with scaler
                    self.scaler.scale(loss).backward()
            else:
                # Standard training
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # Scale loss for gradient accumulation
                loss = loss / self.grad_accumulation_steps

                # Backward pass
                loss.backward()

            # Update metrics
            total_loss += loss.item() * self.grad_accumulation_steps
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Gradient accumulation: only step optimizer after accumulation steps
            if (batch_idx + 1) % self.grad_accumulation_steps == 0:
                # Gradient clipping (optional)
                if hasattr(self.config, "grad_clip") and self.config.grad_clip > 0:
                    if self.use_amp and self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip
                    )

                # Optimizer step
                if self.use_amp and self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                # Zero gradients
                self.optimizer.zero_grad()

                # Update learning rate scheduler
                if self.scheduler is not None:
                    self.scheduler.step()

        # Handle remaining gradients if not divisible by accumulation steps
        if len(self.train_loader) % self.grad_accumulation_steps != 0:
            # Gradient clipping (optional)
            if hasattr(self.config, "grad_clip") and self.config.grad_clip > 0:
                if self.use_amp and self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )

            # Optimizer step
            if self.use_amp and self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            # Update learning rate scheduler
            if self.scheduler is not None:
                self.scheduler.step()

        # Calculate metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total if total > 0 else 0.0

        return {
            "train_loss": avg_loss,
            "train_accuracy": accuracy,
        }

    def validate(self) -> Dict[str, float]:
        """
        Validate the model with optional mixed precision.

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in self.val_loader:
                # Prepare batch
                inputs, labels = self._prepare_batch(batch)

                # Mixed precision inference
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)
                else:
                    # Standard inference
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                # Update metrics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total if total > 0 else 0.0

        return {
            "val_loss": avg_loss,
            "val_accuracy": accuracy,
        }

    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on a given data loader.

        Args:
            data_loader: DataLoader for evaluation data

        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        # Track predictions for additional metrics
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in data_loader:
                # Prepare batch
                inputs, labels = self._prepare_batch(batch)

                # Mixed precision inference
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)
                else:
                    # Standard inference
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                # Update metrics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Store for additional metrics
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate basic metrics
        avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        # Calculate additional metrics if we have data
        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy,
        }

        # Add precision, recall, F1 if we have predictions and sklearn is available
        if len(all_predictions) > 0:
            try:
                from sklearn.metrics import precision_score, recall_score, f1_score

                precision = precision_score(
                    all_labels, all_predictions, average="binary", zero_division=0
                )
                recall = recall_score(
                    all_labels, all_predictions, average="binary", zero_division=0
                )
                f1 = f1_score(
                    all_labels, all_predictions, average="binary", zero_division=0
                )

                metrics.update(
                    {
                        "precision": precision,
                        "recall": recall,
                        "f1_score": f1,
                    }
                )
            except ImportError:
                # sklearn not available, skip additional metrics
                pass

        return metrics

    def train(self, epochs: Optional[int] = None) -> Dict[str, Any]:
        """
        Train the model with early stopping.

        Args:
            epochs: Number of epochs to train (uses config.epochs if None)

        Returns:
            Dictionary with training results
        """
        if epochs is None:
            epochs = self.config.epochs

        print(f"Starting training for {epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")

        for epoch in range(epochs):
            self.epoch = epoch + 1

            # Train for one epoch
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Combine metrics
            metrics = {**train_metrics, **val_metrics}
            self.history.append(metrics)

            # Print progress
            self._print_epoch_progress(epoch + 1, epochs, metrics)

            # Check for improvement
            if val_metrics["val_accuracy"] > self.best_val_acc:
                self.best_val_acc = val_metrics["val_accuracy"]
                self.patience_counter = 0

                # Save best model
                self._save_checkpoint(is_best=True)
                print(f"  ✓ New best model (val acc: {self.best_val_acc:.2%})")
            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= self.config.patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

        # Load best model
        self._load_best_checkpoint()

        # Final evaluation
        final_metrics = self.validate()

        return {
            "best_val_accuracy": self.best_val_acc,
            "final_metrics": final_metrics,
            "history": self.history,
            "epochs_trained": self.epoch,
        }

    def _print_epoch_progress(
        self, epoch: int, total_epochs: int, metrics: Dict[str, float]
    ):
        """Print training progress for current epoch."""
        print(f"Epoch {epoch}/{total_epochs}:")
        print(
            f"  Train Loss: {metrics['train_loss']:.4f}, Train Acc: {metrics['train_accuracy']:.2%}"
        )
        print(
            f"  Val Loss: {metrics['val_loss']:.4f}, Val Acc: {metrics['val_accuracy']:.2%}"
        )

    def _save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint = {
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_acc": self.best_val_acc,
            "history": self.history,
            "config": self.config,
        }

        if self.processor is not None:
            checkpoint["processor_config"] = self.processor.get_config()

        # Save latest checkpoint
        torch.save(checkpoint, checkpoint_dir / "latest_checkpoint.pt")

        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, checkpoint_dir / "best_checkpoint.pt")

    def _load_best_checkpoint(self):
        """Load the best checkpoint."""
        checkpoint_path = Path("checkpoints") / "best_checkpoint.pt"
        if checkpoint_path.exists():
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.epoch = checkpoint["epoch"]
            self.best_val_acc = checkpoint["best_val_acc"]
            self.history = checkpoint["history"]
            print(f"Loaded best checkpoint from epoch {self.epoch}")

    def save_model(self, model_path: Path, **metadata):
        """
        Save trained model.

        Args:
            model_path: Path to save model
            **metadata: Additional metadata to save
        """
        # Try to use the model's save method if it exists
        if hasattr(self.model, "save"):
            self.model.save(str(model_path), processor=self.processor, **metadata)
        else:
            # Fallback to old method
            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "config": self.config,
                "best_val_acc": self.best_val_acc,
                "history": self.history,
                **metadata,
            }

            if self.processor is not None:
                checkpoint["processor_config"] = self.processor.get_config()

            torch.save(checkpoint, model_path)

        print(f"Model saved to {model_path}")


class TrainingStrategy(ABC):
    """Abstract base class for model-specific training strategies."""

    @abstractmethod
    def create_model(self, config: ModelConfig) -> nn.Module:
        """Create model instance."""
        pass

    def load_model(
        self, checkpoint_path: str, config: ModelConfig, device: str = "cpu"
    ) -> Tuple[nn.Module, Any]:
        """
        Load model and processor from checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint
            config: Model configuration
            device: Device to load model on

        Returns:
            Tuple of (model, processor)
        """
        # Default implementation uses model-specific load methods
        if config.model_type == "cnn":
            from ..models.cnn_model import SimpleCNN

            return SimpleCNN.load(checkpoint_path, device)
        elif config.model_type == "lstm":
            from ..models.lstm_model import LSTMModel

            return LSTMModel.load(checkpoint_path, device)
        elif config.model_type == "transformer":
            from ..models.transformer_model import TransformerModel

            return TransformerModel.load(checkpoint_path, device)
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")

    @abstractmethod
    def create_processor(self, config: ModelConfig) -> Any:
        """Create text processor."""
        pass

    @abstractmethod
    def create_dataset(self, data: List[Dict], processor: Any):
        """Create dataset for training."""
        pass

    @abstractmethod
    def create_trainer(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        config: ModelConfig,
        processor: Any,
    ) -> BaseTrainer:
        """Create trainer instance."""
        pass

    def get_collate_fn(self):
        """Get collate function for DataLoader (optional)."""
        return None
