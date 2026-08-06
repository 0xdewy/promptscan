#!/usr/bin/env python3
"""
Configuration management for PromptScan.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal


@dataclass
class DataConfig:
    """Configuration for data handling."""

    # Data paths
    model_dir: Path = field(default_factory=lambda: Path("models"))
    prompts_path: Path = field(
        default_factory=lambda: Path("data/merged.parquet")
    )  # Canonical data source

    # Data processing
    min_freq: int = 5
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42

    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []

        if self.min_freq < 1:
            errors.append("min_freq must be at least 1")

        if not 0 < self.test_size < 1:
            errors.append("test_size must be between 0 and 1")

        if not 0 < self.val_size < 1:
            errors.append("val_size must be between 0 and 1")

        if self.test_size + self.val_size >= 1:
            errors.append("test_size + val_size must be less than 1")

        return errors

    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        return len(self.validate()) == 0


@dataclass
class ModelConfig:
    """Configuration for model training and inference."""

    # Model type
    model_type: Literal["cnn", "lstm", "transformer", "deberta", "ensemble", "pretrained"] = "cnn"

    # Training parameters
    epochs: int = 20
    batch_size: int = 16
    learning_rate: float = 1e-3
    patience: int = 5  # Early stopping patience

    # Device configuration
    device: Literal["cpu", "cuda", "auto"] = "auto"

    # Performance optimizations
    use_amp: bool = False  # Automatic Mixed Precision
    grad_accumulation_steps: int = 1  # Gradient accumulation
    grad_clip: float = 1.0  # Gradient clipping norm

    # Model-specific parameters
    embedding_dim: int = 64
    num_filters: int = 50
    deberta_model: str = "microsoft/deberta-v3-small"
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.3

    # Transformer-specific
    transformer_model: str = "distilbert-base-uncased"
    max_length: int = 256

    # Loss function configuration
    loss_type: Literal["cross_entropy", "focal"] = "focal"
    focal_gamma: float = 2.0  # Focus parameter for focal loss
    use_class_weights: bool = True  # Automatically compute class weights
    label_smoothing: float = 0.0  # Label smoothing for cross-entropy

    # Ensemble configuration
    voting_strategy: Literal["majority", "weighted", "confidence", "soft"] = "weighted"

    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []

        if self.epochs <= 0:
            errors.append("epochs must be positive")

        if self.batch_size <= 0:
            errors.append("batch_size must be positive")

        if self.learning_rate <= 0:
            errors.append("learning_rate must be positive")

        if self.patience < 0:
            errors.append("patience must be non-negative")

        if self.grad_accumulation_steps <= 0:
            errors.append("grad_accumulation_steps must be positive")

        if self.grad_clip < 0:
            errors.append("grad_clip must be non-negative")

        if self.embedding_dim <= 0:
            errors.append("embedding_dim must be positive")

        if self.num_filters <= 0:
            errors.append("num_filters must be positive")

        if self.hidden_dim <= 0:
            errors.append("hidden_dim must be positive")

        if self.num_layers <= 0:
            errors.append("num_layers must be positive")

        if not 0 <= self.dropout <= 1:
            errors.append("dropout must be between 0 and 1")

        if self.max_length <= 0:
            errors.append("max_length must be positive")

        return errors

    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        return len(self.validate()) == 0


@dataclass
class InferenceConfig:
    """Configuration for inference."""

    # Model paths (base names without extension for safetensors format)
    cnn_model: Path = field(default_factory=lambda: Path("models/cnn_best"))
    lstm_model: Path = field(default_factory=lambda: Path("models/lstm_best"))
    transformer_model: Path = field(
        default_factory=lambda: Path("models/transformer_best")
    )

    # Inference parameters
    batch_size: int = 32
    confidence_threshold: float = 0.5
    max_text_length: int = 1000

    # Output formatting
    show_probabilities: bool = False
    show_confidence: bool = True
    output_format: Literal["text", "json", "csv"] = "text"

    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []

        if self.batch_size <= 0:
            errors.append("batch_size must be positive")

        if not 0 <= self.confidence_threshold <= 1:
            errors.append("confidence_threshold must be between 0 and 1")

        if self.max_text_length <= 0:
            errors.append("max_text_length must be positive")

        return errors

    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        return len(self.validate()) == 0
