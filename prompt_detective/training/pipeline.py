#!/usr/bin/env python3
"""
Unified training pipeline for all model types.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..config import AppConfig, DataConfig, ModelConfig
from .base_trainer import TrainingStrategy
from .data_loader import create_dataloaders, load_data_from_parquet, print_data_stats


def train_model(
    model_type: str,
    data_config: DataConfig,
    model_config: Optional[ModelConfig] = None,
    output_dir: Optional[Path] = None,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Unified training function for all model types.

    Args:
        model_type: Type of model to train ("cnn", "lstm", "transformer")
        data_config: Data configuration
        model_config: Model configuration (uses defaults if None)
        output_dir: Directory to save model (default: models/)

    Returns:
        Tuple of (model, processor, training_results)
    """
    # Create model configuration if not provided
    if model_config is None:
        model_config = ModelConfig(model_type=model_type)
    else:
        model_config.model_type = model_type

    # Validate configurations
    data_errors = data_config.validate()
    if data_errors:
        raise ValueError(f"Data configuration errors: {data_errors}")

    model_errors = model_config.validate()
    if model_errors:
        raise ValueError(f"Model configuration errors: {model_errors}")

    # Set output directory
    if output_dir is None:
        output_dir = data_config.model_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load training strategy
    strategy = get_training_strategy(model_type)

    # Load data
    print("Loading data...")
    train_data, val_data, test_data = load_data_from_parquet(
        data_config.train_path,
        data_config.val_path,
        data_config.test_path,
    )

    # Print data statistics
    print_data_stats(train_data, val_data, test_data)

    # Create processor
    print(f"\nCreating {model_type} processor...")
    processor = strategy.create_processor(model_config)

    # Build vocabulary if needed
    if hasattr(processor, "build_vocab"):
        train_texts = [item["text"] for item in train_data]
        processor.build_vocab(train_texts)
        print(f"Vocabulary size: {len(processor.vocab)}")

    # Create model
    print(f"\nCreating {model_type} model...")
    model = strategy.create_model(model_config)

    # Create datasets and dataloaders
    print("\nCreating dataloaders...")
    collate_fn = strategy.get_collate_fn()
    train_loader, val_loader = create_dataloaders(
        train_data,
        val_data,
        processor,
        batch_size=model_config.batch_size,
        collate_fn=collate_fn,
    )

    # Create trainer
    print("\nCreating trainer...")
    trainer = strategy.create_trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=model_config,
        processor=processor,
    )

    # Train model
    print("\nStarting training...")
    results = trainer.train()

    # Save model
    model_filename = f"{model_type}_best.pt"
    model_path = output_dir / model_filename

    trainer.save_model(
        model_path,
        train_acc=results["final_metrics"].get("train_accuracy", 0),
        val_acc=results["final_metrics"].get("val_accuracy", 0),
        epochs_trained=results["epochs_trained"],
    )

    print("\nTraining completed!")
    print(f"Best validation accuracy: {results['best_val_accuracy']:.2%}")
    print(f"Model saved to: {model_path}")

    return model, processor, results


def get_training_strategy(model_type: str) -> TrainingStrategy:
    """
    Get training strategy for model type.

    Args:
        model_type: Type of model ("cnn", "lstm", "transformer")

    Returns:
        TrainingStrategy instance
    """
    if model_type == "cnn":
        from .strategies.cnn_strategy import CNNTrainingStrategy

        return CNNTrainingStrategy()
    elif model_type == "lstm":
        from .strategies.lstm_strategy import LSTMTrainingStrategy

        return LSTMTrainingStrategy()
    elif model_type == "transformer":
        from .strategies.transformer_strategy import TransformerTrainingStrategy

        return TransformerTrainingStrategy()
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_default_config(model_type: str) -> AppConfig:
    """
    Create default configuration for model type.

    Args:
        model_type: Type of model

    Returns:
        AppConfig with model-specific defaults
    """
    config = AppConfig()
    config.model.model_type = model_type

    # Set model-specific defaults
    if model_type == "transformer":
        config.model.epochs = 3
        config.model.learning_rate = 2e-5
        config.model.batch_size = 16
        config.model.max_length = 128
    elif model_type == "lstm":
        config.model.epochs = 20
        config.model.learning_rate = 1e-3
        config.model.batch_size = 32
        config.model.hidden_dim = 128
        config.model.num_layers = 2
    elif model_type == "cnn":
        config.model.epochs = 20
        config.model.learning_rate = 1e-3
        config.model.batch_size = 32
        config.model.embedding_dim = 64
        config.model.num_filters = 50

    return config


if __name__ == "__main__":
    # Test the training pipeline
    import sys

    if len(sys.argv) > 1:
        model_type = sys.argv[1]
    else:
        model_type = "cnn"

    print(f"Testing training pipeline for {model_type} model...")

    try:
        # Create configuration
        config = create_default_config(model_type)

        # Test loading strategy
        strategy = get_training_strategy(model_type)
        print(f"Successfully loaded {model_type} strategy")

        # Test data loading
        train_data, val_data, _ = load_data_from_parquet(
            config.data.train_path,
            config.data.val_path,
        )
        print(
            f"Loaded {len(train_data)} training samples, {len(val_data)} validation samples"
        )

        print("\n✓ Training pipeline test passed!")

    except Exception as e:
        print(f"\n✗ Training pipeline test failed: {e}")
        import traceback

        traceback.print_exc()
