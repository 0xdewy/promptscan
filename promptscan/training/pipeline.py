#!/usr/bin/env python3
"""
Unified training pipeline for all model types.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from torch.utils.data import DataLoader

from ..config import ModelConfig
from .base_trainer import TrainingStrategy
from .data_loader import (
    TextDataset,
    create_dataloaders,
    print_data_stats,
)


def train_model_from_data(
    model_type: str,
    train_data: List[Dict],
    val_data: List[Dict],
    test_data: List[Dict],
    model_config: Optional[ModelConfig] = None,
    output_dir: Optional[Path] = None,
    resume: bool = False,
    checkpoint_path: Optional[str] = None,
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
        resume: Whether to resume from existing checkpoint
        checkpoint_path: Path to checkpoint for resuming (defaults to output_dir/{model_type}_best)

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

    # Determine checkpoint path for resuming
    if checkpoint_path is None:
        checkpoint_path = str(output_dir / f"{model_type}_best")

    # Load training strategy
    strategy = get_training_strategy(model_type)

    # Print data statistics
    print_data_stats(train_data, val_data, test_data)

    # Handle resume logic
    model = None
    processor = None

    def _get_vocab_size(processor) -> int:
        """Safely get vocabulary size from processor."""
        if hasattr(processor, 'vocab_size'):
            return processor.vocab_size
        elif hasattr(processor, 'vocab'):
            return len(processor.vocab)
        return 0

    def _get_max_token_id(processor) -> int:
        """Safely get max token ID from processor vocab."""
        if hasattr(processor, 'vocab'):
            try:
                return max((v for v in processor.vocab.values() if isinstance(v, int)), default=0)
            except (ValueError, TypeError):
                return 0
        return 0

    def _checkpoint_exists(path: Path) -> bool:
        """Check if checkpoint files exist for a model."""
        safetensors_path = path.with_suffix(".safetensors")
        config_path = path.with_suffix(".config.json")
        return safetensors_path.exists() and config_path.exists()

    if resume and _checkpoint_exists(Path(checkpoint_path)):
        print(f"\n🔄 Resuming training from checkpoint: {checkpoint_path}")
        try:
            # Convert "auto" to actual device
            from ..utils.device import get_device

            actual_device = get_device(model_config.device)
            model, processor = strategy.load_model(
                checkpoint_path, model_config, device=actual_device
            )
            print(f"✓ Loaded {model_type} model from checkpoint")
            print(f"  Checkpoint vocabulary: {_get_vocab_size(processor)} tokens")

        except Exception as e:
            print(f"⚠️  Failed to load checkpoint: {e}")
            print("  Starting fresh training instead")
            resume = False

    # Create processor if not loaded from checkpoint
    if processor is None:
        print(f"\nCreating {model_type} processor...")
        processor = strategy.create_processor(model_config)

        # Build vocabulary if needed
        if hasattr(processor, "build_vocab"):
            train_texts = [item["text"] for item in train_data]
            processor.build_vocab(train_texts)
            print(f"Vocabulary size: {_get_vocab_size(processor)}")

    # Create model if not loaded from checkpoint
    if model is None:
        print(f"\nCreating {model_type} model...")
        model = strategy.create_model(model_config)
    else:
        print(f"\nUsing loaded {model_type} model from checkpoint")

        if hasattr(processor, "build_vocab"):
            train_texts = [item["text"] for item in train_data]
            original_vocab_size = _get_vocab_size(processor)
            processor.build_vocab(train_texts)
            new_vocab_size = _get_vocab_size(processor)
            if new_vocab_size != original_vocab_size:
                print(
                    f"  Extended vocabulary: {original_vocab_size} -> {new_vocab_size} tokens"
                )

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

    # Create trainer - pass resume flag for proper embedding resize on resume
    print("\nCreating trainer...")
    trainer = strategy.create_trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=model_config,
        processor=processor,
        resume=resume,
    )

    # Validate training data before starting
    print("\nValidating training data...")
    if hasattr(processor, "validate_training_data"):
        train_texts = [item["text"] for item in train_data]
        val_texts = [item["text"] for item in val_data]
        validation = processor.validate_training_data(train_texts + val_texts)
        if not validation["valid"]:
            raise RuntimeError(
                f"Training data contains out-of-range tokens! "
                f"Max token ID: {validation['max_id_found']}, "
                f"Embedding size: {validation['embedding_size']}, "
                f"Out of range count: {validation['out_of_range_count']}"
            )
        print(f"  Data validation passed (max token ID: {validation['max_id_found']}, embedding: {validation['embedding_size']})")
    else:
        max_token_id = _get_max_token_id(processor)
        embedding_size = getattr(processor, 'next_id', max_token_id + 1)
        if max_token_id >= embedding_size:
            raise RuntimeError(
                f"Processor configuration error: max_token_id ({max_token_id}) >= embedding_size ({embedding_size})"
            )

    # Train model
    print("\nStarting training...")
    results = trainer.train()

    # Evaluate on test set
    results = _evaluate_test_set(trainer, test_loader, results)

    # Save model
    model_filename = f"{model_type}_best"
    model_path = output_dir / model_filename

    train_acc = 0.0
    if "history" in results and len(results["history"]) > 0:
        train_acc = results["history"][-1].get("train_accuracy", 0.0)

    trainer.save_model(
        model_path,
        train_acc=train_acc,
        val_acc=results["final_metrics"].get("val_accuracy", 0),
        val_f1=results.get("best_val_f1", results["final_metrics"].get("val_f1", 0)),
        test_acc=results.get("test_metrics", {}).get("accuracy", 0),
        epochs_trained=results["epochs_trained"],
        best_threshold=results.get("test_metrics", {}).get("best_threshold", 0.5),
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


def get_training_strategy(model_type: str) -> TrainingStrategy:
    """
    Get training strategy for model type.

    Args:
        model_type: Type of model ("cnn", "lstm", "transformer", "deberta")

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
    elif model_type == "deberta":
        from .strategies.deberta_strategy import DeBERTaTrainingStrategy

        return DeBERTaTrainingStrategy()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
