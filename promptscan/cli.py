#!/usr/bin/env python3
"""
Command-line interface for Safe Prompts.
"""

import os

# Set environment variables to suppress warnings BEFORE any imports
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import logging
import warnings

# Suppress ALL warnings BEFORE any imports
warnings.filterwarnings("ignore")

# Also suppress logging warnings
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import argparse
import sys
from pathlib import Path

from . import __version__, get_default_model_save_path
from .unified_detector import UnifiedDetector
from .utils.colors import Colors


def _display_prediction(
    result, model_type, detector=None, source=None, interactive=False
):
    """Display prediction result with individual model predictions for ensemble."""
    if source:
        print(f"{source}:")

    if model_type == "ensemble" and "individual_predictions" in result:
        if not interactive:
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

            # Color-code model type
            model_display = Colors.colored(model_type_display, Colors.model_color(idx))

            # Color-code prediction
            pred_display = Colors.prediction(pred["prediction"], pred["confidence"])

            if interactive:
                print(f"     {model_display}: {pred_display}")
            else:
                print(f"  - {model_display}: {pred_display}")

        if not interactive:
            print(
                f"\n{Colors.header('Ensemble result:')} "
                f"{Colors.prediction(result['prediction'], result['confidence'])}"
            )
        else:
            print(
                f"   {Colors.header('Ensemble:')} "
                f"{Colors.prediction(result['prediction'], result['confidence'])}"
            )
    else:
        if interactive:
            print(f"   {Colors.prediction(result['prediction'], result['confidence'])}")
        else:
            print(
                f"{Colors.header('Result:')} "
                f"{Colors.prediction(result['prediction'], result['confidence'])}"
            )


def predict_command(args):
    """Handle predict command."""
    print("🔍 Safe Prompts - Prompt Injection Detector")
    print("=" * 60)
    print(f"Loading {args.model_type} model...")

    import time

    load_start = time.time()

    try:
        detector = UnifiedDetector(
            model_type=args.model_type,
            model_path=args.model,
            device=args.device,
            voting_strategy=args.voting_strategy,
            model_dir=args.model_dir,
        )
    except FileNotFoundError as e:
        print(f"\n{Colors.error(f'❌ Error: {e}')}")
        print(f"\n{Colors.info('💡 You can:')}")
        if args.model_type == "ensemble":
            print("   1. Train individual models first:")
            print("      promptscan train --model-type cnn")
            print("      promptscan train --model-type lstm")
            print("      promptscan train --model-type transformer")
            print("   2. Use a single model instead: --model-type cnn|lstm|transformer")
            print("   3. Specify custom model directory: --model-dir /path/to/models")
        else:
            print(f"   1. Train a {args.model_type} model:")
            print(f"      promptscan train --model-type {args.model_type}")
            print("   2. Specify a different model path with --model")
            print("   3. Check if model exists in expected locations")
        sys.exit(1)

    # Calculate load time
    load_time = time.time() - load_start

    # Show detector info
    info = detector.get_info()
    print(f"{Colors.success(f'✅ Model loaded successfully in {load_time:.2f}s')}")
    print()

    if args.model_type == "ensemble":
        print(f"{Colors.header('🤝 Ensemble Model Configuration:')}")
        print(f"   {Colors.info(f'Models: {len(info['models'])}')}")
        print(f"   {Colors.info(f'Voting strategy: {info['voting_strategy']}')}")
        print()
        print(f"   {Colors.header('Individual models:')}")
        for model_info in info["models"]:
            params = f"{model_info['parameters']:,}"
            print(f"     • {model_info['type']:12} - {params:>10} parameters")
    else:
        print("📊 Model Information:")
        print(f"   Type: {info['type']}")
        print(f"   Parameters: {info['parameters']:,}")

    print()
    print("=" * 60)
    print()

    if args.file:
        import os
        import time

        print(f"📁 Analyzing file: {args.file}")
        print("-" * 60)

        try:
            # Check if file exists
            if not os.path.exists(args.file):
                print(f"❌ Error: File not found: {args.file}")
                return

            # Get file info
            file_size = os.path.getsize(args.file)
            file_name = os.path.basename(args.file)

            # Try to import markdown parser for file type detection
            try:
                from .utils.markdown_parser import (
                    get_file_type_display,
                    read_and_parse_file,
                )

                has_markdown_parser = True
            except ImportError:
                has_markdown_parser = False

            # Get file type display
            if has_markdown_parser:
                file_type = get_file_type_display(args.file)
                file_type_icon = "📝" if file_type == "Markdown" else "📄"
                print(f"{file_type_icon} File: {file_name} ({file_type})")
            else:
                print(f"📄 File: {file_name}")

            print(f"📦 Size: {file_size:,} bytes")
            print()

            # Read file (with markdown parsing if needed)
            read_start = time.time()
            if has_markdown_parser:
                text = read_and_parse_file(args.file, use_library=True)
                read_method = "read and parsed"
            else:
                with open(args.file, "r") as f:
                    text = f.read().strip()
                read_method = "read"
            read_time = time.time() - read_start

            char_count = len(text)
            print(f"✅ Successfully {read_method} file")
            print(f"   Characters: {char_count:,}")
            print(f"   Read time: {read_time:.2f} seconds")
            print()

            # Check if content is reasonable size
            if char_count > 50000:
                print("⚠️  Warning: File is very large (>50K characters)")
                print("   Analysis may take longer than usual")
                print()

            print("Starting analysis...")
            analysis_start = time.time()

            # Analyze the text
            result = detector.predict(text)

            # Calculate analysis time
            analysis_time = time.time() - analysis_start

            # Display beautiful results
            print()
            print("=" * 60)
            print("📊 FILE ANALYSIS RESULTS")
            print("=" * 60)
            print(f"📁 File: {args.file}")
            print(f"📄 Size: {file_size:,} bytes ({char_count:,} characters)")
            print(
                f"⏱️  Read time: {read_time:.2f}s | Analysis time: {analysis_time:.2f}s"
            )
            print()

            # Display prediction with individual model results
            _display_prediction(result, args.model_type, detector, source=None)

            # Show content preview
            print("📝 Content Preview (first 200 characters):")
            print("-" * 40)

            preview = text[:200]
            if len(text) > 200:
                preview += "..."

            # Clean up preview for display
            preview = preview.replace("\n", " ").replace("\r", " ").replace("  ", " ")
            if len(preview) > 80:
                # Wrap long lines
                wrapped = []
                while len(preview) > 80:
                    wrapped.append(preview[:80])
                    preview = preview[80:]
                if preview:
                    wrapped.append(preview)
                preview = "\n   ".join(wrapped)

            print(f"   {preview}")
            print("-" * 40)
            print()

            # Additional insights
            if result["prediction"] == "INJECTION":
                if result["confidence"] > 0.9:
                    print(f"{Colors.error('⚠️  High confidence injection detected!')}")
                    print(
                        f"   {Colors.DIM}This file appears to contain prompt injection attempts.{Colors.RESET}"
                    )
                else:
                    print(f"{Colors.warning('⚠️  Potential injection detected')}")
                    print(
                        f"   {Colors.DIM}This file may contain suspicious patterns.{Colors.RESET}"
                    )
            else:
                if result["confidence"] > 0.9:
                    print(f"{Colors.success('✅ High confidence safe content')}")
                    print(
                        f"   {Colors.DIM}This file appears to contain safe and legitimate content.{Colors.RESET}"
                    )
                else:
                    print(f"{Colors.warning('⚠️  Low confidence result')}")
                    print(
                        f"   {Colors.DIM}Consider manual review for important decisions.{Colors.RESET}"
                    )

            print()
            print("=" * 60)

        except IOError as e:
            print(f"{Colors.error(f'❌ Error reading file: {e}')}")
        except Exception as e:
            print(f"{Colors.error(f'❌ Error analyzing file: {e}')}")

    elif args.dir:
        from .detector import analyze_directory

        analyze_directory(detector, args.dir, args.summary, args.verbose)

    elif args.url:
        import time

        import requests

        print(f"🌐 Fetching URL: {args.url}")
        print("-" * 60)

        try:
            # Start timing
            fetch_start = time.time()

            # Fetch URL with timeout
            response = requests.get(args.url, timeout=10)
            response.raise_for_status()

            # Calculate fetch time
            fetch_time = time.time() - fetch_start

            # Get content details
            content_type = response.headers.get("content-type", "unknown").split(";")[0]
            content_length = len(response.content)
            text = response.text.strip()
            char_count = len(text)

            # Display fetch details
            print("✅ Successfully fetched URL")
            print(f"   Content type: {content_type}")
            print(f"   Size: {content_length:,} bytes ({char_count:,} characters)")
            print(f"   Fetch time: {fetch_time:.2f} seconds")
            print()

            # Check if content is reasonable size for analysis
            if char_count > 100000:
                print("⚠️  Warning: Content is very large (>100K characters)")
                print("   Analysis may take longer than usual")
                print()

            print("Starting analysis...")
            analysis_start = time.time()

            # Analyze the text
            result = detector.predict(text)

            # Calculate analysis time
            analysis_time = time.time() - analysis_start

            # Display beautiful results
            print()
            print("=" * 60)
            print("📊 URL ANALYSIS RESULTS")
            print("=" * 60)
            print(f"🌐 URL: {args.url}")
            print(f"📄 Content size: {content_length:,} bytes ({char_count:,} chars)")
            print(
                f"⏱️  Fetch time: {fetch_time:.2f}s | Analysis time: {analysis_time:.2f}s"
            )
            print(f"📋 Content type: {content_type}")
            print()

            # Display prediction with individual model results
            _display_prediction(result, args.model_type, detector, source=None)

            # Show content preview
            print("📝 Content Preview (first 300 characters):")
            print("-" * 40)

            preview = text[:300]
            if len(text) > 300:
                preview += "..."

            # Clean up preview for display
            preview = preview.replace("\n", " ").replace("\r", " ").replace("  ", " ")
            if len(preview) > 80:
                # Wrap long lines
                wrapped = []
                while len(preview) > 80:
                    wrapped.append(preview[:80])
                    preview = preview[80:]
                if preview:
                    wrapped.append(preview)
                preview = "\n   ".join(wrapped)

            print(f"   {preview}")
            print("-" * 40)
            print()

            # Additional insights based on prediction
            if result["prediction"] == "INJECTION":
                if result["confidence"] > 0.9:
                    print(f"{Colors.error('⚠️  High confidence injection detected!')}")
                    print(
                        f"   {Colors.DIM}This content appears to contain prompt injection attempts.{Colors.RESET}"
                    )
                else:
                    print(f"{Colors.warning('⚠️  Potential injection detected')}")
                    print(
                        f"   {Colors.DIM}This content may contain suspicious patterns.{Colors.RESET}"
                    )
            else:
                if result["confidence"] > 0.9:
                    print(f"{Colors.success('✅ High confidence safe content')}")
                    print(
                        f"   {Colors.DIM}This content appears to be safe and legitimate.{Colors.RESET}"
                    )
                else:
                    print(f"{Colors.warning('⚠️  Low confidence result')}")
                    print(
                        f"   {Colors.DIM}Consider manual review for important decisions.{Colors.RESET}"
                    )

            print()
            print("=" * 60)

        except requests.exceptions.Timeout:
            print(f"{Colors.error('❌ Error: Request timed out (10 seconds)')}")
            print(
                f"   {Colors.DIM}The server may be slow or unresponsive.{Colors.RESET}"
            )
        except requests.exceptions.HTTPError as e:
            print(f"{Colors.error(f'❌ HTTP Error: {e.response.status_code}')}")
            print(f"   {Colors.DIM}{e.response.reason}{Colors.RESET}")
        except requests.exceptions.ConnectionError:
            print(f"{Colors.error('❌ Connection Error: Could not connect to server')}")
            print(
                f"   {Colors.DIM}Check the URL and your internet connection.{Colors.RESET}"
            )
        except Exception as e:
            print(f"{Colors.error(f'❌ Error fetching URL: {e}')}")

    elif args.text:
        import time

        print("📝 Analyzing provided text")
        print("-" * 60)

        text = args.text.strip()
        char_count = len(text)

        print(f"📄 Text length: {char_count:,} characters")
        print()

        # Check if text is reasonable size
        if char_count > 10000:
            print("⚠️  Warning: Text is very long (>10K characters)")
            print("   Analysis may take longer than usual")
            print()

        print("Starting analysis...")
        analysis_start = time.time()

        # Analyze the text
        result = detector.predict(text)

        # Calculate analysis time
        analysis_time = time.time() - analysis_start

        # Display beautiful results
        print()
        print("=" * 60)
        print("📊 TEXT ANALYSIS RESULTS")
        print("=" * 60)
        print(f"📝 Text length: {char_count:,} characters")
        print(f"⏱️  Analysis time: {analysis_time:.2f} seconds")
        print()

        # Display prediction with individual model results
        _display_prediction(result, args.model_type, detector, source=None)

        # Show text preview
        print("📝 Text Preview:")
        print("-" * 40)

        preview = text[:150]
        if len(text) > 150:
            preview += "..."

        # Clean up preview for display
        preview = preview.replace("\n", " ").replace("\r", " ").replace("  ", " ")
        if len(preview) > 80:
            # Wrap long lines
            wrapped = []
            while len(preview) > 80:
                wrapped.append(preview[:80])
                preview = preview[80:]
            if preview:
                wrapped.append(preview)
            preview = "\n   ".join(wrapped)

        print(f"   {preview}")
        print("-" * 40)
        print()

        # Additional insights
        if result["prediction"] == "INJECTION":
            if result["confidence"] > 0.9:
                print(f"{Colors.error('⚠️  High confidence injection detected!')}")
                print(
                    f"   {Colors.DIM}This text appears to contain prompt injection attempts.{Colors.RESET}"
                )
            else:
                print(f"{Colors.warning('⚠️  Potential injection detected')}")
                print(
                    f"   {Colors.DIM}This text may contain suspicious patterns.{Colors.RESET}"
                )
        else:
            if result["confidence"] > 0.9:
                print(f"{Colors.success('✅ High confidence safe content')}")
                print(
                    f"   {Colors.DIM}This text appears to be safe and legitimate.{Colors.RESET}"
                )
            else:
                print(f"{Colors.warning('⚠️  Low confidence result')}")
                print(
                    f"   {Colors.DIM}Consider manual review for important decisions.{Colors.RESET}"
                )

        print()
        print("=" * 60)

    else:
        # Interactive mode
        print("🔍 Safe Prompts - Interactive Analysis Mode")
        print("=" * 60)
        print("Enter text to analyze for prompt injection.")
        print("Press Ctrl+D (Unix) or Ctrl+Z (Windows) to exit.")
        print("=" * 60)

        analysis_count = 0
        injection_count = 0

        try:
            while True:
                print()
                text = input("📝 Enter text to analyze: ").strip()
                if not text:
                    print("⚠️  Please enter some text to analyze.")
                    continue

                analysis_count += 1
                char_count = len(text)

                print(f"   Analyzing {char_count:,} characters...")

                # Analyze the text
                result = detector.predict(text)

                # Update counts
                if result["prediction"] == "INJECTION":
                    injection_count += 1

                # Display result with individual model predictions
                print(
                    f"   {Colors.prediction(result['prediction'], result['confidence'])}"
                )
                _display_prediction(
                    result, args.model_type, detector, source=None, interactive=True
                )

                # Show quick preview
                preview = text[:80].replace("\n", " ").replace("\r", " ")
                if len(text) > 80:
                    preview += "..."
                print(f"   📋 Preview: {preview}")

                # Session statistics
                print(
                    f"   📊 Session: {analysis_count} analyzed, {injection_count} injections"
                )
                print("   " + "-" * 40)

        except (EOFError, KeyboardInterrupt):
            print()
            print("=" * 60)
            print("👋 Goodbye!")
            if analysis_count > 0:
                safe_count = analysis_count - injection_count
                injection_pct = (
                    (injection_count / analysis_count * 100)
                    if analysis_count > 0
                    else 0
                )
                safe_pct = (
                    (safe_count / analysis_count * 100) if analysis_count > 0 else 0
                )

                print(f"{Colors.header('📊 Session Summary:')}")
                print(f"   Total analyses: {analysis_count}")
                print(
                    f"   {Colors.error(f'🔴 Injections: {injection_count} ({injection_pct:.1f}%)')}"
                )
                print(
                    f"   {Colors.success(f'🟢 Safe: {safe_count} ({safe_pct:.1f}%)')}"
                )
            print("=" * 60)
            sys.exit(0)


def train_command(args):
    """Handle train command."""

    # Load data from prompts.parquet and create dynamic splits
    import os
    import sys

    import pandas as pd

    from .parquet_store import ParquetDataStore

    # Determine which data source to use
    data_source = args.data_source

    print(f"\n📊 Loading data from: {data_source}")

    # Initialize data store
    store = ParquetDataStore(data_source)

    # Get all prompts to check data size
    all_prompts = store.get_all_prompts()
    print(f"📈 Total prompts in database: {len(all_prompts)}")

    # Get statistics
    stats = store.get_statistics()
    print(
        f"   🔴 Injections: {stats['injections']} ({stats['injection_percentage']:.1f}%)"
    )
    print(f"   🟢 Safe: {stats['safe']} ({stats['safe_percentage']:.1f}%)")

    # Check if we have enough data
    if len(all_prompts) < 100:
        print(f"\n⚠️  Warning: Only {len(all_prompts)} prompts available.")
        print("   Consider adding more data before training.")

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

    # Handle pre-split data or create splits
    if args.use_pre_split:
        print("\n🔀 Using pre-split data...")

        # Load pre-split files
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
        # Create training splits dynamically
        print("\n🔀 Creating training splits (80% train, 10% validation, 10% test)...")
        splits = store.get_training_splits(train_ratio=0.8, val_ratio=0.1)

        train_data = convert_to_training_format(splits["train"])
        val_data = convert_to_training_format(splits["val"])
        test_data = convert_to_training_format(splits["test"])

        print(f"   Training set: {len(train_data)} prompts")
        print(f"   Validation set: {len(val_data)} prompts")
        print(f"   Test set: {len(test_data)} prompts")

    print(f"   Training set: {len(train_data)} prompts")
    print(f"   Validation set: {len(val_data)} prompts")
    print(f"   Test set: {len(test_data)} prompts")

    # Set default model path if not provided
    if args.model is None:
        if args.model_type == "cnn":
            args.model = str(get_default_model_save_path("cnn_best"))
        elif args.model_type == "lstm":
            args.model = str(get_default_model_save_path("lstm_best"))
        elif args.model_type == "transformer":
            args.model = str(get_default_model_save_path("transformer_best"))

    print(f"\n🏋️  Training {args.model_type} model...")
    print(f"  Model will be saved to: {args.model}")
    print(
        f"  Training parameters: {args.epochs} epochs, batch size {args.batch_size}, lr {args.learning_rate}"
    )

    # Use the new training pipeline that accepts data directly
    from .config import ModelConfig
    from .training.pipeline import train_model_from_data

    # Create model configuration
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

    # Train model using the new pipeline that accepts data directly
    model, processor, results = train_model_from_data(
        model_type=args.model_type,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        model_config=model_config,
        output_dir=Path(args.model).parent,
        resume=args.resume,
        checkpoint_path=args.model if args.resume else None,
    )

    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"Best validation accuracy: {results['best_val_accuracy']:.2%}")

    if "test_metrics" in results:
        test_metrics = results["test_metrics"]
        print(f"Test accuracy: {test_metrics['accuracy']:.2%}")

        # Show additional metrics if available
        if "precision" in test_metrics:
            print(f"Test precision: {test_metrics['precision']:.2%}")
            print(f"Test recall: {test_metrics['recall']:.2%}")
            print(f"Test F1-score: {test_metrics['f1_score']:.2%}")

        # Compare validation vs test
        val_vs_test_diff = abs(results["best_val_accuracy"] - test_metrics["accuracy"])
        if val_vs_test_diff > 0.05:  # More than 5% difference
            print(
                f"⚠️  Note: Validation-test gap: {val_vs_test_diff:.2%} (may indicate overfitting)"
            )
        else:
            print(f"✓ Validation-test consistency: {val_vs_test_diff:.2%}")
    else:
        print("⚠️  Test evaluation not available (no test data provided)")

    print("=" * 60)


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

            with importlib.resources.path("promptscan", "data") as data_dir:
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


def insert_command(args):
    """Handle insert command - add new prompts to the database."""
    from .parquet_store import ParquetDataStore

    # Use output file if specified, otherwise use parquet file
    parquet_file = args.output if args.output else args.parquet

    # Initialize the data store
    store = ParquetDataStore(parquet_file)

    # Check if we're in batch mode (GitHub, directory, or file list)
    if args.github or args.dir or (args.file and len(args.file) > 0):
        return _batch_insert_command(args, store)
    else:
        return _interactive_insert_command(args, store)


def _batch_insert_command(args, store):
    """Handle batch insert from GitHub, directory, or file list."""
    from .batch_importer import BatchImporter

    print("🔧 Safe Prompts - Batch Insertion Mode")
    print("=" * 60)

    # Parse extensions
    extensions = None
    if args.extensions:
        extensions = [ext.strip() for ext in args.extensions.split(",")]
        extensions = [ext if ext.startswith(".") else f".{ext}" for ext in extensions]
        print(f"📁 Extensions: {', '.join(extensions)}")

    # Parse max size
    max_size = None
    if args.max_size:
        max_size = _parse_size(args.max_size)
        print(f"📏 Max size: {_format_size(max_size)}")

    # Parse label
    label = None
    if args.label:
        label = args.label == "injection"
        print(f"🏷️  Label: {args.label}")

    # Check if label is required but not provided
    if (
        args.github or args.dir or (args.file and len(args.file) > 1)
    ) and not args.label:
        print(
            "\n❌ Error: --label is required for batch imports from GitHub, directories, or multiple files."
        )
        print("   Use --label safe or --label injection")
        return

    # Initialize batch importer
    importer = BatchImporter(
        store, verbose=args.verbose, github_token=args.github_token
    )

    # Process based on source
    stats = None
    if args.github:
        print(f"🌐 Source: GitHub repository - {args.github}")
        print(f"🌿 Branch: {args.branch}")
        print()

        stats = importer.import_from_github(
            github_url=args.github,
            label=label,
            extensions=extensions,
            exclude=args.exclude,
            max_size=max_size,
            max_depth=10,
            dry_run=args.dry_run,
            github_token=args.github_token,
            branch=args.branch,
        )

    elif args.dir:
        print(f"📂 Source: Local directory - {args.dir}")
        print()

        stats = importer.import_from_directory(
            directory=args.dir,
            label=label,
            extensions=extensions,
            exclude=args.exclude,
            max_size=max_size,
            recursive=True,
            dry_run=args.dry_run,
        )

    elif args.file and len(args.file) > 0:
        print(f"📄 Source: {len(args.file)} file(s)")
        print()

        stats = importer.import_from_files(
            files=args.file, label=label, dry_run=args.dry_run
        )

    # Show results
    if stats:
        print("\n" + "=" * 60)
        print("📊 IMPORT COMPLETE")
        print("=" * 60)
        print(str(stats))

        if not args.dry_run:
            # Show database statistics
            db_stats = store.get_statistics()
            print(f"\n💾 Database now contains {db_stats['total']} prompts:")
            print(
                f"  🔴 {db_stats['injections']} injection prompts ({db_stats['injection_percentage']:.1f}%)"
            )
            print(
                f"  🟢 {db_stats['safe']} safe prompts ({db_stats['safe_percentage']:.1f}%)"
            )

        if args.dry_run:
            print("\n⚠️  DRY RUN - No changes were made to the database.")

    print("\n✅ Done!")


def _interactive_insert_command(args, store):
    """Handle interactive insert mode (original behavior)."""
    print("💬 Safe Prompts - Interactive Insertion Mode")
    print("=" * 50)
    print("Enter prompts to add to the database.")
    print("Press Ctrl+D (Unix) or Ctrl+Z (Windows) to exit.")
    print()

    try:
        while True:
            # Get prompt text
            text = input("Enter prompt text: ").strip()
            if not text:
                print("Prompt text cannot be empty. Please try again.")
                continue

            # Get label
            while True:
                label_input = (
                    input("Is this an injection prompt? (y/n): ").strip().lower()
                )
                if label_input in ["y", "yes"]:
                    is_injection = True
                    break
                elif label_input in ["n", "no"]:
                    is_injection = False
                    break
                else:
                    print("Please enter 'y' for injection or 'n' for safe prompt.")

            # Add to database
            prompt_id = store.add_prompt(text, is_injection)
            if prompt_id is None:
                label_type = "injection" if is_injection else "safe"
                print(f"⚠️  Prompt already exists in database as {label_type} prompt")
                print()
                continue

            label_type = "injection" if is_injection else "safe"
            print(f"✓ Added prompt #{prompt_id} as {label_type} prompt")
            print()

            # Ask if user wants to continue
            while True:
                continue_input = input("Add another prompt? (y/n): ").strip().lower()
                if continue_input in ["y", "yes"]:
                    print()
                    break
                elif continue_input in ["n", "no"]:
                    # Show statistics
                    stats = store.get_statistics()
                    print(f"\nDatabase now contains {stats['total']} prompts:")
                    print(
                        f"  🔴 {stats['injections']} injection prompts ({stats['injection_percentage']:.1f}%)"
                    )
                    print(
                        f"  🟢 {stats['safe']} safe prompts ({stats['safe_percentage']:.1f}%)"
                    )
                    return
                else:
                    print("Please enter 'y' or 'n'.")

    except (EOFError, KeyboardInterrupt):
        # Show statistics
        stats = store.get_statistics()
        print(f"\n\nDatabase now contains {stats['total']} prompts:")
        print(
            f"  🔴 {stats['injections']} injection prompts ({stats['injection_percentage']:.1f}%)"
        )
        print(f"  🟢 {stats['safe']} safe prompts ({stats['safe_percentage']:.1f}%)")
        print("\n👋 Goodbye!")


def _parse_size(size_str: str) -> int:
    """Parse size string like '1MB', '500KB', '1000000' to bytes."""
    size_str = size_str.strip().upper()

    # Check for unit suffixes
    if size_str.endswith("KB"):
        return int(float(size_str[:-2]) * 1024)
    elif size_str.endswith("MB"):
        return int(float(size_str[:-2]) * 1024 * 1024)
    elif size_str.endswith("GB"):
        return int(float(size_str[:-2]) * 1024 * 1024 * 1024)
    else:
        # Assume bytes
        return int(size_str)


def _format_size(size_bytes: int) -> str:
    """Format bytes to human readable string."""
    if size_bytes >= 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
    elif size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes} bytes"


def import_command(args):
    """Handle import command - import prompts from a parquet file."""
    from pathlib import Path

    import pandas as pd

    from .parquet_store import ParquetDataStore

    # Check if source file exists
    source_path = Path(args.source)
    if not source_path.exists():
        print(f"Error: Source file not found: {args.source}")
        return

    # Load source data
    try:
        source_df = pd.read_parquet(source_path)
        print(f"Loaded {len(source_df)} prompts from {args.source}")
    except Exception as e:
        print(f"Error loading parquet file: {e}")
        return

    # Check required columns
    required_columns = {"text", "is_injection"}
    if not required_columns.issubset(source_df.columns):
        print(f"Error: Source file must contain columns: {required_columns}")
        print(f"Found columns: {source_df.columns.tolist()}")
        return

    # Initialize target database
    store = ParquetDataStore(args.target)
    before_stats = store.get_statistics()

    print("\nTarget database before import:")
    print(f"  Total prompts: {before_stats['total']}")
    print(
        f"  Injection prompts: {before_stats['injections']} ({before_stats['injection_percentage']:.1f}%)"
    )
    print(
        f"  Safe prompts: {before_stats['safe']} ({before_stats['safe_percentage']:.1f}%)"
    )

    # Import the data
    print(f"\nImporting {len(source_df)} prompts...")
    try:
        store.import_from_dataframe(source_df)
    except Exception as e:
        print(f"Error importing data: {e}")
        return

    # Get updated stats
    after_stats = store.get_statistics()
    print("\nTarget database after import:")
    print(f"  Total prompts: {after_stats['total']}")
    print(
        f"  Injection prompts: {after_stats['injections']} ({after_stats['injection_percentage']:.1f}%)"
    )
    print(
        f"  Safe prompts: {after_stats['safe']} ({after_stats['safe_percentage']:.1f}%)"
    )

    added = after_stats["total"] - before_stats["total"]
    print(f"\nSuccessfully added {added} new prompts to {args.target}")


def hf_download_command(args):
    """Handle HF download command."""
    try:
        from .hf_utils import download_all_models_from_hf
    except ImportError:
        print(
            "Error: huggingface-hub not installed. Install with: pip install huggingface-hub"
        )
        sys.exit(1)

    print("Downloading models from Hugging Face Hub...")
    print(f"Repository: {args.repo_id}")
    print(f"Output directory: {args.output_dir}")

    success = download_all_models_from_hf(
        repo_id=args.repo_id,
        output_dir=args.output_dir,
        token=args.token,
        force_download=args.force,
    )

    if success:
        print(f"\n✓ All models downloaded successfully to {args.output_dir}/")
        print("  - cnn_best.safetensors + .config.json")
        print("  - lstm_best.safetensors + .config.json")
        print("  - transformer_best.safetensors + .config.json")
        print('\nYou can now use: promptscan predict "Your text here"')
    else:
        print("\n⚠ Some models failed to download")
        sys.exit(1)


def hf_upload_command(args):
    """Handle HF upload command."""
    try:
        from huggingface_hub import HfApi, create_repo, upload_file
        from huggingface_hub.utils import HfHubHTTPError
    except ImportError:
        print(
            "Error: huggingface-hub not installed. Install with: pip install huggingface-hub"
        )
        sys.exit(1)

    print("Uploading models to Hugging Face Hub...")
    print(f"Repository: {args.repo_id}")
    print(f"Model directory: {args.model_dir}")

    # Get token from args or environment
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN environment variable not set and no token provided")
        print("Set HF_TOKEN in your environment or pass --token argument")
        sys.exit(1)

    api = HfApi(token=token)

    # Check if repository exists, create if not
    try:
        repo_info = api.repo_info(repo_id=args.repo_id, repo_type="model")
        print(f"Repository exists: {repo_info.id}")
    except HfHubHTTPError as e:
        if e.status_code == 404:
            print(f"Creating repository: {args.repo_id}")
            try:
                create_repo(
                    repo_id=args.repo_id,
                    token=token,
                    private=args.private,
                    repo_type="model",
                    exist_ok=True,
                )
                print(f"Created repository: {args.repo_id}")
            except Exception as create_error:
                print(f"Error creating repository: {create_error}")
                sys.exit(1)
        else:
            print(f"Error checking repository: {e}")
            sys.exit(1)

    # Upload each model type
    model_types = ["cnn", "lstm", "transformer"]
    success_count = 0

    for model_type in model_types:
        print(f"\n{'=' * 60}")
        print(f"Uploading {model_type.upper()} model...")
        print(f"{'=' * 60}")

        # Model files
        base_name = f"{model_type}_best"
        safetensors_file = Path(args.model_dir) / f"{base_name}.safetensors"
        config_file = Path(args.model_dir) / f"{base_name}.config.json"

        if not safetensors_file.exists():
            print(f"  Warning: {safetensors_file} not found, skipping")
            continue

        if not config_file.exists():
            print(f"  Warning: {config_file} not found, skipping")
            continue

        # Create model directory in repo
        repo_model_dir = model_type

        try:
            # Upload safetensors file
            print(f"  Uploading weights: {safetensors_file.name}")
            upload_file(
                path_or_fileobj=str(safetensors_file),
                path_in_repo=f"{repo_model_dir}/model.safetensors",
                repo_id=args.repo_id,
                token=token,
                commit_message=f"{args.commit_message} - {model_type} weights",
            )

            # Upload config file
            print(f"  Uploading config: {config_file.name}")
            upload_file(
                path_or_fileobj=str(config_file),
                path_in_repo=f"{repo_model_dir}/config.json",
                repo_id=args.repo_id,
                token=token,
                commit_message=f"{args.commit_message} - {model_type} config",
            )

            # Also upload with original names for compatibility
            upload_file(
                path_or_fileobj=str(safetensors_file),
                path_in_repo=f"{repo_model_dir}/{base_name}.safetensors",
                repo_id=args.repo_id,
                token=token,
                commit_message=f"{args.commit_message} - {model_type} weights (original name)",
            )

            upload_file(
                path_or_fileobj=str(config_file),
                path_in_repo=f"{repo_model_dir}/{base_name}.config.json",
                repo_id=args.repo_id,
                token=token,
                commit_message=f"{args.commit_message} - {model_type} config (original name)",
            )

            print(f"  ✓ {model_type.upper()} model uploaded successfully")
            success_count += 1

        except Exception as e:
            print(f"  ✗ Error uploading {model_type} model: {e}")

    # Upload README and model card if they exist
    print(f"\n{'=' * 60}")
    print("Uploading documentation...")
    print(f"{'=' * 60}")

    try:
        # Check for README.md
        readme_file = Path("README.md")
        if readme_file.exists():
            print("  Uploading README.md")
            upload_file(
                path_or_fileobj=str(readme_file),
                path_in_repo="README.md",
                repo_id=args.repo_id,
                token=token,
                commit_message=f"{args.commit_message} - README",
            )

        # Check for model card
        model_card_file = Path("model_card.md")
        if model_card_file.exists():
            print("  Uploading model_card.md")
            upload_file(
                path_or_fileobj=str(model_card_file),
                path_in_repo="model_card.md",
                repo_id=args.repo_id,
                token=token,
                commit_message=f"{args.commit_message} - model card",
            )

        print("  ✓ Documentation uploaded successfully")

    except Exception as e:
        print(f"  ✗ Error uploading documentation: {e}")

    print(f"\n{'=' * 60}")
    print("Upload Summary")
    print(f"{'=' * 60}")
    print(f"Repository: {args.repo_id}")
    print(f"Models uploaded: {success_count}/{len(model_types)}")
    print(f"View at: https://huggingface.co/{args.repo_id}")

    if success_count == len(model_types):
        print("\n✓ All models uploaded successfully!")
    else:
        print(f"\n⚠ Only {success_count}/{len(model_types)} models uploaded")
        sys.exit(1)


def hf_list_command(args):
    """Handle HF list command."""
    try:
        from .hf_utils import check_hf_model_available, get_hf_model_info
    except ImportError:
        print(
            "Error: huggingface-hub not installed. Install with: pip install huggingface-hub"
        )
        sys.exit(1)

    print(f"Checking Hugging Face Hub repository: {args.repo_id}")

    # Check if repo exists
    repo_info = get_hf_model_info(repo_id=args.repo_id, token=args.token)

    if repo_info:
        print("\n✓ Repository found:")
        print(f"  ID: {repo_info['id']}")
        print(f"  Last modified: {repo_info['last_modified']}")
        print(f"  Private: {repo_info['private']}")
        if "tags" in repo_info and repo_info["tags"]:
            print(f"  Tags: {', '.join(repo_info['tags'])}")
        if "downloads" in repo_info:
            print(f"  Downloads: {repo_info['downloads']}")
        if "likes" in repo_info:
            print(f"  Likes: {repo_info['likes']}")

        # Check for individual models
        print("\nChecking for individual models...")
        model_types = ["cnn", "lstm", "transformer"]

        for model_type in model_types:
            available = check_hf_model_available(
                repo_id=args.repo_id,
                model_dir=model_type,
                token=args.token,
            )
            status = "✓ Available" if available else "✗ Not found"
            print(f"  {model_type.upper()}: {status}")

        print("\nTo download models: promptscan hf download")

    else:
        print("\n✗ Repository not found or inaccessible")
        print(f"Check: https://huggingface.co/{args.repo_id}")
        sys.exit(1)


def convert_command(args):
    """Handle convert command."""
    from .convert_model import convert_directory, convert_pt_to_safetensors

    input_path = Path(args.input)

    if args.batch or input_path.is_dir():
        success = convert_directory(args.input, args.output, args.force)
    else:
        success = convert_pt_to_safetensors(args.input, args.output, args.force)

    if not success:
        sys.exit(1)


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
    # Analyze text with beautiful output
    promptscan predict "Ignore all previous instructions"

    # Analyze file with details
    promptscan predict --file input.txt

    # Analyze directory with overview
    promptscan predict --dir ./prompts

    # Analyze directory with verbose file list
    promptscan predict --dir ./prompts --verbose

    # Analyze URL with fetch details
    promptscan predict --url https://example.com

    # Train model
    promptscan train

    # Export data
    promptscan export --format json --output prompts.json

    # Show version
    promptscan --version
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
        "--dir", "-d", help="Analyze all text files in directory (.txt, .md, .markdown)"
    )
    predict_parser.add_argument("--url", "-u", help="Analyze text from URL")
    predict_parser.add_argument(
        "--summary", action="store_true", help="Show summary for directory analysis"
    )
    predict_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show verbose output with details"
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
    train_parser.add_argument(
        "--data-source",
        default="data/prompts.parquet",
        help="Data source for training (default: data/prompts.parquet)",
    )
    train_parser.add_argument(
        "--use-pre-split",
        action="store_true",
        help="Use pre-split data (train_split.parquet, val_split.parquet, test_split.parquet)",
    )
    train_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from existing checkpoint if available",
    )
    train_parser.set_defaults(func=train_command)

    # Convert command
    convert_parser = subparsers.add_parser(
        "convert-model", help="Convert .pt model files to safetensors format"
    )
    convert_parser.add_argument(
        "input", help="Input .pt file or directory containing .pt files"
    )
    convert_parser.add_argument(
        "-o", "--output", help="Output file or directory (default: same as input)"
    )
    convert_parser.add_argument(
        "-f", "--force", action="store_true", help="Overwrite existing files"
    )
    convert_parser.add_argument(
        "--batch", action="store_true", help="Batch convert all .pt files in directory"
    )
    convert_parser.set_defaults(func=convert_command)

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

    # Insert command
    insert_parser = subparsers.add_parser(
        "insert", help="Insert new prompts into the database from various sources"
    )
    insert_parser.add_argument(
        "--parquet", default="data/prompts.parquet", help="Target parquet file path"
    )

    # Source arguments (mutually exclusive groups)
    source_group = insert_parser.add_mutually_exclusive_group()
    source_group.add_argument("--github", "-g", help="GitHub repository URL")
    source_group.add_argument("--dir", "-d", help="Local directory path")
    source_group.add_argument("--file", "-f", action="append", help="File path(s)")

    # Filter arguments
    insert_parser.add_argument(
        "--extensions",
        help="Comma-separated file extensions to include (e.g., .md,.txt,.py)",
    )
    insert_parser.add_argument(
        "--exclude",
        action="append",
        help="Path patterns to exclude (can be used multiple times)",
    )
    insert_parser.add_argument(
        "--max-size", help="Maximum file size (e.g., 1MB, 500KB, 1000000)"
    )

    # Batch mode arguments
    insert_parser.add_argument(
        "--batch",
        "-b",
        action="store_true",
        help="Batch mode (no interactive confirmations)",
    )
    insert_parser.add_argument(
        "--label",
        "-l",
        choices=["safe", "injection"],
        help="Label for all items in batch mode",
    )
    insert_parser.add_argument(
        "--dry-run", action="store_true", help="Preview without importing"
    )
    insert_parser.add_argument(
        "--output", "-o", help="Output parquet file (default: use --parquet)"
    )

    # GitHub-specific arguments
    insert_parser.add_argument(
        "--github-token", help="GitHub personal access token (for higher rate limits)"
    )
    insert_parser.add_argument(
        "--branch", default="main", help="Git branch (default: main)"
    )

    # Verbose output
    insert_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show verbose output"
    )

    insert_parser.set_defaults(func=insert_command)

    # Import command
    import_parser = subparsers.add_parser(
        "import", help="Import prompts from a parquet file"
    )
    import_parser.add_argument("source", help="Source parquet file to import from")
    import_parser.add_argument(
        "--target", default="data/prompts.parquet", help="Target parquet file path"
    )
    import_parser.set_defaults(func=import_command)

    # Hugging Face Hub commands
    hf_parser = subparsers.add_parser(
        "hf", help="Hugging Face Hub operations (download/upload models)"
    )
    hf_subparsers = hf_parser.add_subparsers(
        title="hf commands", dest="hf_command", help="Available HF commands"
    )

    # Download models from HF
    download_parser = hf_subparsers.add_parser(
        "download", help="Download models from Hugging Face Hub"
    )
    download_parser.add_argument(
        "--repo-id", default="0xdewy/promptscan", help="Hugging Face repository ID"
    )
    download_parser.add_argument(
        "--output-dir", default="models", help="Directory to save downloaded models"
    )
    download_parser.add_argument(
        "--token",
        help="Hugging Face token (for private repos, default: HF_TOKEN env var)",
    )
    download_parser.add_argument(
        "--force", action="store_true", help="Force re-download even if cached"
    )
    download_parser.set_defaults(func=hf_download_command)

    # Upload models to HF
    upload_parser = hf_subparsers.add_parser(
        "upload", help="Upload models to Hugging Face Hub (requires write access)"
    )
    upload_parser.add_argument(
        "--repo-id", default="0xdewy/promptscan", help="Hugging Face repository ID"
    )
    upload_parser.add_argument(
        "--model-dir", default="models", help="Directory containing model files"
    )
    upload_parser.add_argument(
        "--token", help="Hugging Face token (default: HF_TOKEN env var)"
    )
    upload_parser.add_argument(
        "--private", action="store_true", help="Create private repository"
    )
    upload_parser.add_argument(
        "--commit-message", default="Upload promptscan models", help="Commit message"
    )
    upload_parser.set_defaults(func=hf_upload_command)

    # List available models on HF
    list_parser = hf_subparsers.add_parser(
        "list", help="List available models on Hugging Face Hub"
    )
    list_parser.add_argument(
        "--repo-id", default="0xdewy/promptscan", help="Hugging Face repository ID"
    )
    list_parser.add_argument("--token", help="Hugging Face token (for private repos)")
    list_parser.set_defaults(func=hf_list_command)

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
