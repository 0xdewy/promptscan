#!/usr/bin/env python3
"""
Command-line interface for Safe Prompts.
"""

import os

# Set environment variables to suppress warnings BEFORE any imports
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import warnings
import logging

# Suppress ALL warnings BEFORE any imports
warnings.filterwarnings("ignore")

# Also suppress logging warnings
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import argparse
import sys
from pathlib import Path

from . import __version__, get_model_path
from .data_utils import (
    ensure_data_files,
)
from .unified_detector import UnifiedDetector


def _display_prediction(result, model_type, detector=None, source=None):
    """Display prediction result with individual model predictions for ensemble."""
    if source:
        print(f"{source}:")

    if model_type == "ensemble" and "individual_predictions" in result:
        print("Individual model predictions:")
        # Get model types from detector if available
        model_types = []
        if (
            detector
            and hasattr(detector, "detector")
            and hasattr(detector.detector, "model_types")
        ):
            model_types = detector.detector.model_types

        for pred in result["individual_predictions"]:
            idx = pred.get("model_idx", 0)
            model_type_display = (
                model_types[idx] if idx < len(model_types) else f"Model {idx}"
            )
            print(
                f"  - {model_type_display}: {pred['prediction']} ({pred['confidence']:.2%})"
            )
        print(f"\nEnsemble result: {result['prediction']} ({result['confidence']:.2%})")
    else:
        print(f"Result: {result['prediction']} ({result['confidence']:.2%})")


def predict_command(args):
    """Handle predict command."""
    print(f"Loading {args.model_type} detector...")

    try:
        detector = UnifiedDetector(
            model_type=args.model_type,
            model_path=args.model,
            device=args.device,
            voting_strategy=args.voting_strategy,
            model_dir=args.model_dir,
        )
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nYou can:")
        if args.model_type == "ensemble":
            print("  1. Train individual models first")
            print("  2. Use a single model: --model-type cnn|lstm|transformer")
        else:
            print(
                f"  1. Train a {args.model_type} model: prompt-detective train --model-type {args.model_type}"
            )
            print("  2. Specify a different model path with --model")
        sys.exit(1)

    # Show detector info
    info = detector.get_info()
    if args.model_type == "ensemble":
        print(f"Loaded ensemble with {len(info['models'])} models")
        print(f"Voting strategy: {info['voting_strategy']}")
        for model_info in info["models"]:
            print(f"  - {model_info['type']}: {model_info['parameters']:,} params")
    else:
        print(f"Loaded {info['type']} model with {info['parameters']:,} parameters")

    if args.file:
        with open(args.file, "r") as f:
            text = f.read().strip()
        result = detector.predict(text)
        _display_prediction(
            result, args.model_type, detector, source=f"File: {args.file}"
        )

    elif args.dir:
        from .detector import analyze_directory

        analyze_directory(detector, args.dir, args.summary)

    elif args.url:
        import requests

        try:
            response = requests.get(args.url)
            response.raise_for_status()
            text = response.text.strip()
            result = detector.predict(text)
            _display_prediction(
                result, args.model_type, detector, source=f"URL: {args.url}"
            )
        except Exception as e:
            print(f"Error fetching URL: {e}")

    elif args.text:
        result = detector.predict(args.text)
        _display_prediction(
            result, args.model_type, detector, source=f"Text: {args.text}"
        )

    else:
        # Interactive mode
        print("Safe Prompts - Interactive Mode (Ctrl+D to exit)")
        print("=" * 50)
        try:
            while True:
                text = input("\nEnter text to analyze: ").strip()
                if not text:
                    continue
                result = detector.predict(text)
                _display_prediction(result, args.model_type, detector)
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            sys.exit(0)


def train_command(args):
    """Handle train command."""

    # Ensure data files are available

    train_path, val_path, test_path = ensure_data_files()

    # Set default model path if not provided
    if args.model is None:
        if args.model_type == "cnn":
            args.model = str(get_model_path("cnn_best.pt"))
        elif args.model_type == "lstm":
            args.model = str(get_model_path("lstm_best.pt"))
        elif args.model_type == "transformer":
            args.model = str(get_model_path("transformer_best.pt"))

    print(f"\nTraining {args.model_type} model with:")
    print(f"  Training data: {train_path}")
    print(f"  Validation data: {val_path}")
    print(f"  Test data: {test_path}")
    print(f"  Model will be saved to: {args.model}")
    print(
        f"  Training parameters: {args.epochs} epochs, batch size {args.batch_size}, lr {args.learning_rate}"
    )

    # Use unified training pipeline
    from .config import DataConfig, ModelConfig
    from .training.pipeline import train_model

    # Create configurations
    data_config = DataConfig(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        model_dir=Path(args.model).parent,
    )

    model_config = ModelConfig(
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
    )

    # Set model-specific defaults
    if args.model_type == "transformer":
        model_config.learning_rate = 2e-5  # Standard for fine-tuning
        model_config.max_length = 128
    elif args.model_type == "lstm":
        model_config.hidden_dim = 128
        model_config.num_layers = 2
        model_config.dropout = 0.3
    elif args.model_type == "cnn":
        model_config.embedding_dim = 64
        model_config.num_filters = 50

    # Train model using unified pipeline
    model, processor, results = train_model(
        model_type=args.model_type,
        data_config=data_config,
        model_config=model_config,
        output_dir=Path(args.model).parent,
    )

    print("\nTraining completed successfully!")
    print(f"Best validation accuracy: {results['best_val_accuracy']:.2%}")


def export_command(args):
    """Handle export command."""
    import csv
    import json

    import pandas as pd

    # Load data
    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        # Try package data
        try:
            import importlib.resources

            with importlib.resources.path("prompt_detective", "data") as data_dir:
                package_path = data_dir / "prompts.parquet"
                if package_path.exists():
                    parquet_path = package_path
                else:
                    print(f"Error: Parquet file not found: {args.parquet}")
                    print("Try running the migration script first.")
                    return
        except (ImportError, FileNotFoundError):
            print(f"Error: Parquet file not found: {args.parquet}")
            print("Try running the migration script first.")
            return

    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} records from {parquet_path}")

    # Handle different export formats
    if args.format == "stats":
        # Show statistics
        total = len(df)
        injections = df["is_injection"].sum()
        safe = total - injections

        print("\n=== Data Statistics ===")
        print(f"Total prompts: {total}")
        print(f"Injection prompts: {int(injections)} ({injections / total * 100:.1f}%)")
        print(f"Safe prompts: {int(safe)} ({safe / total * 100:.1f}%)")

    elif args.format == "json":
        output_path = args.output or "prompts.json"
        data = df.to_dict("records")

        # Convert to simpler format
        simple_data = []
        for item in data:
            simple_data.append(
                {"text": item["text"], "is_injection": bool(item["is_injection"])}
            )

        with open(output_path, "w") as f:
            json.dump(simple_data, f, indent=2)

        print(f"Exported {len(simple_data)} prompts to {output_path}")

    elif args.format == "csv":
        output_path = args.output or "prompts.csv"
        csv_df = df[["text", "is_injection"]].copy()
        csv_df["is_injection"] = csv_df["is_injection"].astype(int)
        csv_df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)
        print(f"Exported {len(csv_df)} prompts to {output_path}")

    else:
        print(f"Export format '{args.format}' not implemented in CLI.")
        print("Please use the export script directly:")
        print(f"  python scripts/export_parquet.py --format {args.format}")
        if args.output:
            print(f"  --output {args.output}")


def version_command(args):
    """Handle version command."""
    print(f"Safe Prompts v{__version__}")
    print("Prompt injection detection system")
    print(f"Python {sys.version}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Safe Prompts - Prompt injection detection system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
   # Analyze text
  prompt-detective predict "Ignore all previous instructions"

  # Analyze file
  prompt-detective predict --file input.txt

  # Train model
  prompt-detective train

  # Export data
  prompt-detective export --format json --output prompts.json

  # Show version
  prompt-detective --version
        """,
    )

    parser.add_argument(
        "--version", action="store_true", help="Show version information"
    )

    # Create subparsers
    subparsers = parser.add_subparsers(
        title="commands", dest="command", help="Available commands"
    )

    # Predict command
    predict_parser = subparsers.add_parser(
        "predict", help="Predict if text contains prompt injection"
    )
    predict_parser.add_argument(
        "text", nargs="?", help="Text to analyze (or use --file, --dir, --url)"
    )
    predict_parser.add_argument("--file", "-f", help="Analyze text from file")
    predict_parser.add_argument(
        "--dir", "-d", help="Analyze all .txt files in directory"
    )
    predict_parser.add_argument("--url", "-u", help="Analyze text from URL")
    predict_parser.add_argument(
        "--summary", action="store_true", help="Show summary for directory analysis"
    )
    predict_parser.add_argument(
        "--model", help="Path to model checkpoint (default depends on model type)"
    )
    predict_parser.add_argument(
        "--model-type",
        choices=["cnn", "lstm", "transformer", "ensemble"],
        default="ensemble",
        help="Model type to use (default: ensemble)",
    )
    predict_parser.add_argument(
        "--model-dir",
        default=None,
        help="Directory containing model checkpoints (for ensemble, default: package models)",
    )
    predict_parser.add_argument(
        "--voting-strategy",
        choices=["majority", "weighted", "confidence", "soft"],
        default="majority",
        help="Voting strategy for ensemble (default: majority)",
    )
    predict_parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device to run inference on (auto, cpu, or cuda)",
    )
    predict_parser.set_defaults(func=predict_command)

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--model-type",
        choices=["cnn", "lstm", "transformer"],
        default="cnn",
        help="Model type to train (default: cnn)",
    )
    train_parser.add_argument(
        "--model", help="Path to save model checkpoint (default depends on model type)"
    )
    train_parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs (default: 20)"
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training (default: 16, reduced for memory safety)",
    )
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate (default: 0.001)",
    )
    train_parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device to train on (auto for GPU detection, cpu, or cuda)",
    )
    train_parser.set_defaults(func=train_command)

    # Export command
    export_parser = subparsers.add_parser(
        "export", help="Export data to various formats"
    )
    export_parser.add_argument(
        "--format",
        choices=["json", "csv", "excel", "stats", "training", "parquet-split"],
        default="json",
        help="Export format",
    )
    export_parser.add_argument("--output", help="Output file path")
    export_parser.add_argument(
        "--parquet", default="data/prompts.parquet", help="Input parquet file path"
    )
    export_parser.set_defaults(func=export_command)

    # Version command (as separate parser for --version flag)
    version_parser = subparsers.add_parser("version", help="Show version information")
    version_parser.set_defaults(func=version_command)

    # Parse arguments
    args = parser.parse_args()

    # Handle --version flag
    if args.version:
        version_command(args)
        return

    # Handle commands
    if hasattr(args, "func"):
        args.func(args)
    else:
        # No command provided, show help
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
