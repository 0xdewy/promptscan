#!/usr/bin/env python3
"""Train PromptScan models."""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from promptscan.config import ModelConfig
from promptscan.parquet_store import ParquetDataStore
from promptscan.training.pipeline import train_model_from_data


def convert_to_training_format(df):
    """Convert DataFrame with 'is_injection' to list of dicts with 'label'."""
    texts = df["text"].tolist()
    labels = df["is_injection"].astype(int).tolist()
    optional_fields = ["source", "category", "variation_type"]
    optional_data = {
        field: df[field].tolist() if field in df.columns else None
        for field in optional_fields
    }
    records = []
    for i in range(len(texts)):
        record = {"text": texts[i], "label": labels[i]}
        for field in optional_fields:
            data = optional_data[field]
            if data is not None and pd.notna(data[i]):
                record[field] = data[i]
        records.append(record)
    return records


def train_command(args):
    """Handle train command."""
    data_source = args.data_source

    print(f"\n📊 Loading data from: {data_source}")

    store = ParquetDataStore(data_source)
    all_prompts = store.get_all_prompts()
    print(f"📈 Total prompts in database: {len(all_prompts)}")

    stats = store.get_statistics()
    print(
        f"   🔴 Injections: {stats['injections']} ({stats['injection_percentage']:.1f}%)"
    )
    print(f"   🟢 Safe: {stats['safe']} ({stats['safe_percentage']:.1f}%)")

    if len(all_prompts) < 100:
        print(f"\n⚠️  Warning: Only {len(all_prompts)} prompts available.")
        print("   Consider adding more data before training.")

    if args.max_samples > 0 and args.max_samples < len(all_prompts):
        print(
            f"\n📊 Using random subset of {args.max_samples:,} samples (from {len(all_prompts):,} total)"
        )

    if args.use_pre_split:
        print("\n🔀 Using pre-split data...")

        train_path = "data/train_split.parquet"
        val_path = "data/val_split.parquet"
        test_path = "data/test_split.parquet"

        if not os.path.exists(train_path):
            print(f"❌ Error: Pre-split training file not found: {train_path}")
            print(
                "   Run the data unification script first: python scripts/unify_data.py --create-splits"
            )
            sys.exit(1)

        train_df = pd.read_parquet(train_path)
        val_df = pd.read_parquet(val_path)
        test_df = pd.read_parquet(test_path)

        train_data = convert_to_training_format(train_df)
        val_data = convert_to_training_format(val_df)
        test_data = convert_to_training_format(test_df)

        print(f"   Training set: {len(train_data)} prompts (pre-split)")
        print(f"   Validation set: {len(val_data)} prompts (pre-split)")
        print(f"   Test set: {len(test_data)} prompts (pre-split)")
    else:
        print("\n🔀 Creating training splits (80% train, 10% validation, 10% test)...")
        splits = store.get_training_splits(
            train_ratio=0.8,
            val_ratio=0.1,
            max_samples=args.max_samples,
            max_samples_per_source=args.max_samples_per_source,
        )

        train_data = convert_to_training_format(splits["train"])
        val_data = convert_to_training_format(splits["val"])
        test_data = convert_to_training_format(splits["test"])

        print(f"   Training set: {len(train_data)} prompts")
        print(f"   Validation set: {len(val_data)} prompts")
        print(f"   Test set: {len(test_data)} prompts")

    model_path = None
    if args.model is None:
        if args.model_type == "cnn":
            model_path = str(Path("models") / "cnn_best")
        elif args.model_type == "lstm":
            model_path = str(Path("models") / "lstm_best")
        elif args.model_type == "transformer":
            model_path = str(Path("models") / "transformer_best")
        elif args.model_type == "deberta":
            model_path = str(Path("models") / "deberta_best")

    print(f"\n🏋️  Training {args.model_type} model...")
    print(f"  Model will be saved to: {model_path}")
    print(
        f"  Training parameters: {args.epochs} epochs, batch size {args.batch_size}, lr {args.learning_rate}"
    )

    model_config = ModelConfig(
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        use_amp=args.amp if hasattr(args, "amp") else False,
        loss_type="focal",
        focal_gamma=2.0,
        use_class_weights=True,
    )

    if args.model_type == "transformer":
        model_config.learning_rate = 2e-5
        model_config.max_length = 128
    elif args.model_type == "deberta":
        model_config.learning_rate = 2e-5
        model_config.max_length = 512
        model_config.use_amp = True
    elif args.model_type == "lstm":
        model_config.hidden_dim = 128
        model_config.num_layers = 2
        model_config.dropout = 0.3
    elif args.model_type == "cnn":
        model_config.embedding_dim = 64
        model_config.num_filters = 50

    resume = not args.new
    model, processor, results = train_model_from_data(
        model_type=args.model_type,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        model_config=model_config,
        output_dir=Path(model_path).parent if model_path else Path("models"),
        resume=resume,
        checkpoint_path=model_path if resume else None,
    )

    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"Best validation accuracy: {results['best_val_accuracy']:.2%}")

    if "test_metrics" in results:
        test_metrics = results["test_metrics"]
        print(f"Test accuracy: {test_metrics['accuracy']:.2%}")

        if "precision" in test_metrics:
            print(f"Test precision: {test_metrics['precision']:.2%}")
            print(f"Test recall: {test_metrics['recall']:.2%}")
            print(f"Test F1-score: {test_metrics['f1_score']:.2%}")

        val_vs_test_diff = abs(results["best_val_accuracy"] - test_metrics["accuracy"])
        if val_vs_test_diff > 0.05:
            print(
                f"⚠️  Note: Validation-test gap: {val_vs_test_diff:.2%} (may indicate overfitting)"
            )
        else:
            print(f"✓ Validation-test consistency: {val_vs_test_diff:.2%}")
    else:
        print("⚠️  Test evaluation not available (no test data provided)")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Train PromptScan models")
    parser.add_argument(
        "--model-type",
        choices=["cnn", "lstm", "transformer", "deberta"],
        default="cnn",
        help="Model type to train (default: cnn)",
    )
    parser.add_argument(
        "--model", help="Path to save model checkpoint (default depends on model type)"
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs (default: 20)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training (default: 16)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device to train on",
    )
    parser.add_argument(
        "--data-source",
        default="data/merged.parquet",
        help="Data source for training (default: data/merged.parquet)",
    )
    parser.add_argument(
        "--use-pre-split", action="store_true", help="Use pre-split data"
    )
    parser.add_argument(
        "--new",
        action="store_true",
        help="Start fresh training (default: resume if checkpoint exists)",
    )
    parser.add_argument(
        "--amp", action="store_true", help="Enable automatic mixed precision (AMP)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Maximum number of samples (0 = use all)",
    )
    parser.add_argument(
        "--max-samples-per-source",
        type=int,
        default=0,
        help="Cap samples per source to reduce source dominance (0 = no cap, recommended: 10000)",
    )

    args = parser.parse_args()
    train_command(args)


if __name__ == "__main__":
    main()
