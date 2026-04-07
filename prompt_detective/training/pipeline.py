#!/usr/bin/env python3
"""
Unified training pipeline for all model types.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from ..config import AppConfig, DataConfig, ModelConfig
from ..parquet_store import ParquetDataStore
from .base_trainer import TrainingStrategy
from .data_loader import (
    create_dataloaders,
    load_data_from_parquet,
    print_data_stats,
    TextDataset,
)


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

    # Load data from prompts.parquet with dynamic splits
    print(f"Loading data from: {data_config.prompts_path}")

    # Initialize data store
    store = ParquetDataStore(data_config.prompts_path)

    # Get all prompts to check data size
    all_prompts = store.get_all_prompts()
    print(f"Total prompts in database: {len(all_prompts)}")

    # Get statistics
    stats = store.get_statistics()
    print(
        f"   Injections: {stats['injections']} ({stats['injection_percentage']:.1f}%)"
    )
    print(f"   Safe: {stats['safe']} ({stats['safe_percentage']:.1f}%)")

    # Check if we have enough data
    if len(all_prompts) < 100:
        print(f"\nWarning: Only {len(all_prompts)} prompts available.")
        print("   Consider adding more data before training.")

    # Create training splits
    train_ratio = 1.0 - data_config.test_size - data_config.val_size
    print(
        f"\nCreating training splits ({train_ratio * 100:.0f}% train, {data_config.val_size * 100:.0f}% validation, {data_config.test_size * 100:.0f}% test)..."
    )
    splits = store.get_training_splits(
        train_ratio=train_ratio,
        val_ratio=data_config.val_size,
    )

    # Convert to list of dictionaries for training pipeline
    # Training pipeline expects "label" field (1 for injection, 0 for safe)
    # Parquet store has "is_injection" field (True/False)
    import pandas as pd

    def convert_to_training_format(df):
        """Convert DataFrame with 'is_injection' to list of dicts with 'label'."""
        records = []
        for _, row in df.iterrows():
            # Create base record with required fields
            record = {"text": row["text"], "label": 1 if row["is_injection"] else 0}

            # Add optional fields if present
            optional_fields = ["source", "category", "variation_type"]
            for field in optional_fields:
                if field in df.columns and pd.notna(row[field]):
                    record[field] = row[field]

            records.append(record)
        return records

    train_data = convert_to_training_format(splits["train"])
    val_data = convert_to_training_format(splits["val"])
    test_data = convert_to_training_format(splits["test"])

    # Print split statistics (more detailed than print_data_stats)
    print(f"\nSplit Statistics:")
    print(f"  Training: {len(train_data)} samples")
    print(f"  Validation: {len(val_data)} samples")
    print(f"  Test: {len(test_data)} samples")
    print(f"  Total: {len(train_data) + len(val_data) + len(test_data)} samples")

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

    # Create test loader if test data is available
    test_loader = _create_test_loader(
        test_data, processor, model_config.batch_size, collate_fn
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

    # Evaluate on test set
    results = _evaluate_test_set(trainer, test_loader, results)

    # Save model
    model_filename = f"{model_type}_best.pt"
    model_path = output_dir / model_filename

    trainer.save_model(
        model_path,
        train_acc=results["final_metrics"].get("train_accuracy", 0),
        val_acc=results["final_metrics"].get("val_accuracy", 0),
        test_acc=results.get("test_metrics", {}).get("accuracy", 0),
        epochs_trained=results["epochs_trained"],
    )

    print("\nTraining completed!")
    print(f"Best validation accuracy: {results['best_val_accuracy']:.2%}")
    if "test_metrics" in results:
        print(f"Test accuracy: {results['test_metrics']['accuracy']:.2%}")
    print(f"Model saved to: {model_path}")

    return model, processor, results


def train_model_from_data(
    model_type: str,
    train_data: List[Dict],
    val_data: List[Dict],
    test_data: List[Dict],
    model_config: Optional[ModelConfig] = None,
    output_dir: Optional[Path] = None,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Unified training function that accepts data directly (not from files).

    Args:
        model_type: Type of model to train ("cnn", "lstm", "transformer")
        train_data: Training data as list of dictionaries
        val_data: Validation data as list of dictionaries
        test_data: Test data as list of dictionaries
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

    # Validate model configuration
    model_errors = model_config.validate()
    if model_errors:
        raise ValueError(f"Model configuration errors: {model_errors}")

    # Set output directory
    if output_dir is None:
        output_dir = Path("models")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load training strategy
    strategy = get_training_strategy(model_type)

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

    # Create test loader if test data is available
    test_loader = _create_test_loader(
        test_data, processor, model_config.batch_size, collate_fn
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

    # Evaluate on test set
    results = _evaluate_test_set(trainer, test_loader, results)

    # Save model
    model_filename = f"{model_type}_best.pt"
    model_path = output_dir / model_filename

    trainer.save_model(
        model_path,
        train_acc=results["final_metrics"].get("train_accuracy", 0),
        val_acc=results["final_metrics"].get("val_accuracy", 0),
        test_acc=results.get("test_metrics", {}).get("accuracy", 0),
        epochs_trained=results["epochs_trained"],
    )

    print("\nTraining completed!")
    print(f"Best validation accuracy: {results['best_val_accuracy']:.2%}")
    if "test_metrics" in results:
        print(f"Test accuracy: {results['test_metrics']['accuracy']:.2%}")
    print(f"Model saved to: {model_path}")

    return model, processor, results


def _create_test_loader(test_data, processor, batch_size, collate_fn):
    """Create test data loader if test data is available."""
    if not test_data:
        return None

    test_dataset = TextDataset(test_data, processor)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )
    return test_loader


def _evaluate_test_set(trainer, test_loader, results):
    """Evaluate model on test set and update results."""
    if not test_loader:
        return results

    print("\n" + "=" * 60)
    print("Evaluating on Test Set")
    print("=" * 60)

    test_metrics = trainer.evaluate(test_loader)

    # Add test metrics to results
    results["test_metrics"] = test_metrics

    print(f"Test Accuracy: {test_metrics['accuracy']:.2%}")
    if "precision" in test_metrics:
        print(f"Test Precision: {test_metrics['precision']:.2%}")
        print(f"Test Recall: {test_metrics['recall']:.2%}")
        print(f"Test F1-Score: {test_metrics['f1_score']:.2%}")

    return results


def create_default_config(model_type: str = "cnn") -> AppConfig:
    """
    Create default configuration for training.

    Args:
        model_type: Type of model ("cnn", "lstm", "transformer")

    Returns:
        AppConfig instance with default settings
    """
    # Create model config with specified type
    model_config = ModelConfig(model_type=model_type)

    # Create data config with default paths
    data_config = DataConfig()

    # Create app config combining both
    app_config = AppConfig(model=model_config, data=data_config)

    return app_config


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


if __name__ == "__main__":
    # Test the training pipeline
    try:
        # Create test configuration
        data_config = DataConfig()
        model_config = ModelConfig(model_type="cnn", epochs=1, batch_size=4)

        print("Testing training pipeline...")
        model, processor, results = train_model(
            model_type="cnn",
            data_config=data_config,
            model_config=model_config,
            output_dir=Path("test_models"),
        )

        print("\n✓ Training pipeline test passed!")
        print(f"Best validation accuracy: {results['best_val_accuracy']:.2%}")

        # Clean up test directory
        import shutil

        if Path("test_models").exists():
            shutil.rmtree("test_models")

    except Exception as e:
        print(f"\n✗ Training pipeline test failed: {e}")
        import traceback

        traceback.print_exc()
