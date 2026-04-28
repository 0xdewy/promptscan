#!/usr/bin/env python3
"""
Configuration management for Safe Prompts.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional


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
class InferenceConfig:
    """Configuration for inference."""

    # Model paths (base names without extension for safetensors format)
    cnn_model: Path = field(default_factory=lambda: Path("models/best_model"))
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


@dataclass
class AppConfig:
    """Application configuration."""

    # Model configuration
    model: ModelConfig = field(default_factory=ModelConfig)

    # Data configuration
    data: DataConfig = field(default_factory=DataConfig)

    # Inference configuration
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    # Application settings
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    cache_dir: Path = field(default_factory=lambda: Path(".cache"))

    def validate(self) -> List[str]:
        """Validate all configurations."""
        errors = []

        errors.extend([f"model.{err}" for err in self.model.validate()])
        errors.extend([f"data.{err}" for err in self.data.validate()])
        errors.extend([f"inference.{err}" for err in self.inference.validate()])

        return errors

    def is_valid(self) -> bool:
        """Check if all configurations are valid."""
        return len(self.validate()) == 0

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "AppConfig":
        """Create configuration from dictionary."""
        model_dict = config_dict.get("model", {})
        data_dict = config_dict.get("data", {})
        inference_dict = config_dict.get("inference", {})

        if "cache_dir" in config_dict and isinstance(config_dict["cache_dir"], str):
            config_dict["cache_dir"] = Path(config_dict["cache_dir"])

        return cls(
            model=ModelConfig(**model_dict),
            data=DataConfig(**data_dict),
            inference=InferenceConfig(**inference_dict),
            debug=config_dict.get("debug", False),
            log_level=config_dict.get("log_level", "INFO"),
            cache_dir=config_dict.get("cache_dir", Path(".cache")),
        )

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            "model": {
                "model_type": self.model.model_type,
                "epochs": self.model.epochs,
                "batch_size": self.model.batch_size,
                "learning_rate": self.model.learning_rate,
                "patience": self.model.patience,
                "device": self.model.device,
                "use_amp": self.model.use_amp,
                "grad_accumulation_steps": self.model.grad_accumulation_steps,
                "grad_clip": self.model.grad_clip,
                "embedding_dim": self.model.embedding_dim,
                "num_filters": self.model.num_filters,
                "hidden_dim": self.model.hidden_dim,
                "num_layers": self.model.num_layers,
                "dropout": self.model.dropout,
                "transformer_model": self.model.transformer_model,
                "max_length": self.model.max_length,
                "voting_strategy": self.model.voting_strategy,
            },
            "data": {
                "model_dir": self.data.model_dir,
                "prompts_path": self.data.prompts_path,
                "min_freq": self.data.min_freq,
                "test_size": self.data.test_size,
                "val_size": self.data.val_size,
                "random_state": self.data.random_state,
            },
            "inference": {
                "cnn_model": self.inference.cnn_model,
                "lstm_model": self.inference.lstm_model,
                "transformer_model": self.inference.transformer_model,
                "batch_size": self.inference.batch_size,
                "confidence_threshold": self.inference.confidence_threshold,
                "max_text_length": self.inference.max_text_length,
                "show_probabilities": self.inference.show_probabilities,
                "show_confidence": self.inference.show_confidence,
                "output_format": self.inference.output_format,
            },
            "debug": self.debug,
            "log_level": self.log_level,
            "cache_dir": self.cache_dir,
        }


def load_config(config_path: Optional[Path] = None) -> AppConfig:
    """
    Load configuration from file or use defaults.

    Args:
        config_path: Path to configuration file (JSON or YAML)

    Returns:
        AppConfig instance
    """
    if config_path and config_path.exists():
        import json

        with open(config_path, "r") as f:
            config_dict = json.load(f)

        config = AppConfig.from_dict(config_dict)
    else:
        # Use default configuration
        config = AppConfig()

    # Validate configuration
    errors = config.validate()
    if errors:
        raise ValueError(f"Configuration errors: {errors}")

    return config


def save_config(config: AppConfig, config_path: Path):
    """
    Save configuration to file.

    Args:
        config: AppConfig instance
        config_path: Path to save configuration file
    """
    config_dict = config.to_dict()

    import json

    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)


# Default configuration instance
default_config = AppConfig()


if __name__ == "__main__":
    # Test the configuration system
    config = AppConfig()

    print("Default configuration:")
    print(f"Model type: {config.model.model_type}")
    print(f"Epochs: {config.model.epochs}")
    print(f"Batch size: {config.model.batch_size}")
    print(f"Learning rate: {config.model.learning_rate}")

    # Test validation
    print(f"\nConfiguration valid: {config.is_valid()}")

    # Test invalid configuration
    invalid_config = AppConfig(model=ModelConfig(epochs=-1, batch_size=0))
    errors = invalid_config.validate()
    print(f"\nInvalid configuration errors: {errors}")

    # Test dictionary conversion
    config_dict = config.to_dict()
    print(f"\nConfiguration as dictionary keys: {list(config_dict.keys())}")

    # Test loading from dictionary
    loaded_config = AppConfig.from_dict(config_dict)
    print(f"\nLoaded configuration model type: {loaded_config.model.model_type}")
