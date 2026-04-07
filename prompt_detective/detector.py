#!/usr/bin/env python3
"""
Simple Prompt Injection Detector
A clean, minimal neural network for detecting prompt injections.
"""

import argparse
import os
import pickle
import sys
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from . import get_model_path
from .data_utils import get_default_data_paths
from .models.cnn_model import SimpleCNN
from .processors.word_processor import WordProcessor
from .training.data_loader import load_data_from_parquet
from .utils.data_loader import PromptDataset
from .utils.device import get_device


class SimpleTrainer:
    """Minimal trainer for the model."""

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        processor,
        device="cpu",
        learning_rate=1e-3,
    ):
        self.device = get_device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.processor = processor

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in self.train_loader:
            inputs = batch["input_ids"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model({"input_ids": inputs})
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        return total_loss / len(self.train_loader), correct / total

    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in self.val_loader:
                inputs = batch["input_ids"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model({"input_ids": inputs})
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return total_loss / len(self.val_loader), correct / total

    def train(self, epochs=20, patience=3):
        """Train the model with early stopping."""
        best_val_acc = 0
        patience_counter = 0

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            print(f"Epoch {epoch + 1}/{epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "vocab": self.processor.vocab,
                        "max_length": self.processor.max_length,
                        "vocab_size": self.model.embedding.num_embeddings,
                        "val_acc": val_acc,
                        "epoch": epoch + 1,
                    },
                    str(get_model_path("best_model.pt")),
                )
                print(f"  ✓ Saved best model (val_acc: {val_acc:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break


class SimplePromptDetector:
    """Main class for prompt injection detection.

    DEPRECATED: Use UnifiedDetector instead for ensemble models and better API.
    This class is kept for backward compatibility only.
    """

    def __init__(self, model_path=None, device="cpu"):
        import warnings

        warnings.warn(
            "SimplePromptDetector is deprecated. Use UnifiedDetector instead for ensemble models and better API.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.device = get_device(device)
        if model_path is None:
            model_path = str(get_model_path("best_model.pt"))
        self.load_model(model_path)

    def load_model(self, model_path):
        """Load trained model and processor."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Try to load with weights_only=True first (safer)
        try:
            checkpoint = torch.load(
                model_path, map_location=self.device, weights_only=True
            )
        except (pickle.UnpicklingError, RuntimeError):
            # If that fails, try with weights_only=False
            # (for old models with processor objects)
            import warnings

            warnings.warn(
                f"Loading model with weights_only=False - ensure {model_path} "
                "is from a trusted source",
                stacklevel=2,
            )
            checkpoint = torch.load(
                model_path, map_location=self.device, weights_only=False
            )

        # Handle both old and new model formats
        if "processor" in checkpoint:
            # Old format: processor object is saved directly (pre-PyTorch 2.6)
            self.processor = checkpoint["processor"]
            vocab_size = checkpoint.get("vocab_size", len(self.processor.vocab))
        elif "vocab" in checkpoint:
            # New format: vocab dictionary is saved (safer, PyTorch 2.6+ compatible)
            from .processors.word_processor import WordProcessor

            self.processor = WordProcessor(
                max_length=checkpoint.get("max_length", 100),
                min_freq=checkpoint.get("min_freq", 2),
            )
            self.processor.vocab = checkpoint["vocab"]
            # Create inverse vocab mapping
            self.processor.inverse_vocab = {
                v: k for k, v in checkpoint["vocab"].items()
            }
            self.processor.next_id = len(checkpoint["vocab"])
            vocab_size = checkpoint.get("vocab_size", len(checkpoint["vocab"]))
        else:
            raise KeyError("Checkpoint must contain either 'processor' or 'vocab' key")

        self.model = SimpleCNN(vocab_size)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> Dict[str, Any]:
        """Predict if text contains prompt injection."""
        encoded = self.processor.encode(text)
        input_tensor = encoded["input_ids"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            pred_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][int(pred_class)].item()

        return {
            "prediction": "INJECTION" if pred_class == 1 else "SAFE",
            "confidence": confidence,
            "class": pred_class,
            "probabilities": probabilities[0].cpu().numpy().tolist(),
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Predict for multiple texts."""
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results


def train_model(
    train_path="data/train.parquet",
    val_path="data/val.parquet",
    test_path="data/test.parquet",
    model_path=None,
    epochs=20,
    batch_size=32,
    learning_rate=1e-3,
    device="cpu",
):
    """Train the model from parquet files.

    DEPRECATED: Use `promptscan train` command instead which uses
    prompts.parquet with dynamic splits.
    """
    import warnings

    warnings.warn(
        "train_model is deprecated. Use 'promptscan train' command instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Convert device string to actual device
    device = get_device(device)

    # Setup memory-safe training
    try:
        from ..utils.memory_monitor import setup_memory_safe_training

        batch_size, memory_monitor = setup_memory_safe_training(
            batch_size, max_memory_mb=8000
        )
    except ImportError:
        print("WARNING: Memory monitor not available, using default batch size")
        memory_monitor = None

    print("Loading data from parquet files...")
    train_data, val_data, test_data = load_data_from_parquet(
        train_path, val_path, test_path
    )

    print(f"Training data: {len(train_data)} samples")
    print(f"Validation data: {len(val_data)} samples")
    print(f"Test data: {len(test_data)} samples")

    # Build processor
    processor = WordProcessor(max_length=100, min_freq=2)
    train_texts = [item["text"] for item in train_data]
    processor.build_vocab(train_texts)

    print(f"Vocabulary size: {len(processor.vocab)}")

    # Create datasets
    train_dataset = PromptDataset(train_data, processor)
    val_dataset = PromptDataset(val_data, processor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    model = SimpleCNN(vocab_size=len(processor.vocab))

    # Train
    trainer = SimpleTrainer(
        model,
        train_loader,
        val_loader,
        processor,
        device=device,
        learning_rate=learning_rate,
    )
    trainer.train(epochs=epochs)

    # Save model
    if model_path is None:
        model_path = str(get_model_path("best_model.pt"))
    model.save(
        model_path,
        processor,
        train_acc=trainer.train_epoch()[1],
        val_acc=trainer.validate()[1],
    )

    print(f"Model saved to {model_path}")
    print(f"Vocabulary size: {len(processor.vocab)}")


def legacy_main():
    """Legacy CLI entry point (kept for backward compatibility)."""
    parser = argparse.ArgumentParser(description="Prompt Injection Detector")
    parser.add_argument(
        "text",
        nargs="?",
        help="Text to analyze (or use --file, --dir, --url)",
    )
    parser.add_argument(
        "--train", action="store_true", help="Train the model from parquet files"
    )
    parser.add_argument("--file", "-f", help="Analyze text from file")
    parser.add_argument("--dir", "-d", help="Analyze all .txt files in directory")
    parser.add_argument("--url", "-u", help="Analyze text from URL")
    parser.add_argument(
        "--summary", action="store_true", help="Show summary for directory analysis"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to model checkpoint (default: package model)",
    )

    args = parser.parse_args()

    if args.train:
        print(
            "⚠️  DEPRECATED: The --train option in SimplePromptDetector is deprecated."
        )
        print("   Use 'promptscan train' command instead for modern training.")
        print("   The modern training:")
        print("   1. Uses prompts.parquet as the data source")
        print("   2. Creates dynamic splits (80% train, 10% validation, 10% test)")
        print("   3. Includes batch-imported data from GitHub/docs")
        print("   4. Supports ensemble models (CNN, LSTM, Transformer)")
        print()
        print("   To train with the modern pipeline:")
        print("   $ promptscan train --model-type ensemble")
        print()

        # Try to use the old training for backward compatibility
        try:
            train_path, val_path, test_path = get_default_data_paths()
            train_model(
                train_path=str(train_path),
                val_path=str(val_path),
                test_path=str(test_path),
                model_path=args.model,
            )
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
            print(
                "   Static split files (train.parquet, val.parquet, test.parquet) not found."
            )
            print("   These have been replaced by prompts.parquet with dynamic splits.")
            print("   Run 'promptscan train' instead.")
        return

    # Initialize detector
    try:
        detector = SimplePromptDetector(model_path=args.model)
    except FileNotFoundError:
        print(f"Error: Model file '{args.model}' not found.")
        print("Train a model first with: python detector.py --train")
        sys.exit(1)

    # Handle different input types
    if args.file:
        with open(args.file, "r") as f:
            text = f.read().strip()
        result = detector.predict(text)
        print(f"File: {args.file}")
        print(f"Result: {result['prediction']} ({result['confidence']:.2%})")

    elif args.dir:
        analyze_directory(detector, args.dir, args.summary, verbose=False)

    elif args.url:
        import requests

        try:
            response = requests.get(args.url)
            response.raise_for_status()
            text = response.text.strip()
            result = detector.predict(text)
            print(f"URL: {args.url}")
            print(f"Result: {result['prediction']} ({result['confidence']:.2%})")
        except Exception as e:
            print(f"Error fetching URL: {e}")

    elif args.text:
        result = detector.predict(args.text)
        print(f"Text: {args.text}")
        print(f"Result: {result['prediction']} ({result['confidence']:.2%})")

    else:
        parser.print_help()


def analyze_directory(detector, directory_path, show_summary=False, verbose=False):
    """Analyze all text files (.txt, .md, .markdown) in directory with beautiful output."""
    import glob
    import time
    from pathlib import Path

    # Try to import markdown parser
    try:
        from .utils.markdown_parser import read_and_parse_file, get_file_type_display

        has_markdown_parser = True
    except ImportError:
        has_markdown_parser = False

    # Start timing
    start_time = time.time()

    # Find all text files (.txt, .md, .markdown)
    text_files = []
    for ext in [".txt", ".md", ".markdown"]:
        text_files.extend(glob.glob(os.path.join(directory_path, f"*{ext}")))

    if not text_files:
        print(f"📁 No text files (.txt, .md, .markdown) found in {directory_path}")
        return

    # Sort files for consistent display
    text_files.sort()

    # Display directory overview
    print(f"📁 Scanning directory: {directory_path}")
    print(f"📄 Found {len(text_files)} text file{'s' if len(text_files) != 1 else ''}")
    print()

    if verbose:
        print("Files to analyze:")
        for i, file_path in enumerate(text_files, 1):
            file_size = os.path.getsize(file_path)
            size_str = f"{file_size:,} bytes"
            if file_size > 1024:
                size_str = f"{file_size / 1024:.1f} KB"

            # Add file type indicator
            if has_markdown_parser:
                file_type = get_file_type_display(file_path)
                file_display = f"{os.path.basename(file_path)} ({file_type})"
            else:
                file_display = os.path.basename(file_path)

            print(f"  {i:3d}. {file_display:40} ({size_str})")
        print()

    print("Starting analysis...")
    print("-" * 60)

    results = []
    for i, file_path in enumerate(text_files, 1):
        # Read file (with markdown parsing if needed)
        if has_markdown_parser:
            text = read_and_parse_file(file_path, use_library=True)
        else:
            with open(file_path, "r") as f:
                text = f.read().strip()

        # Analyze
        result = detector.predict(text)

        # Store results
        file_size = os.path.getsize(file_path)
        results.append(
            {
                "file": os.path.basename(file_path),
                "path": file_path,
                "text": text,
                "size": file_size,
                "prediction": result["prediction"],
                "confidence": result["confidence"],
                "is_injection": result["prediction"] == "INJECTION",
            }
        )

        # Display progress
        file_name = os.path.basename(file_path)
        if len(file_name) > 25:
            file_name = file_name[:22] + "..."

        # Get file type display
        if has_markdown_parser:
            file_type = get_file_type_display(file_name)
            if file_type == "Markdown":
                file_type_icon = "📝"
            elif file_type == "Text":
                file_type_icon = "📄"
            else:
                file_type_icon = "📎"
            file_display = f"{file_type_icon} {file_name}"
        else:
            file_display = f"📄 {file_name}"

        # Color and icon based on prediction
        if result["prediction"] == "INJECTION":
            icon = "🔴"
            status = "INJECTION"
        else:
            icon = "🟢"
            status = "SAFE"

        # Progress indicator
        progress = f"[{i}/{len(text_files)}]"

        # File size indicator
        size_str = f"{file_size:,}B"
        if file_size > 1024:
            size_str = f"{file_size / 1024:.1f}KB"

        print(
            f"{progress} {icon} {file_display:28} {status:10} ({result['confidence']:.1%}) {size_str:>8}"
        )

    # Calculate total time
    total_time = time.time() - start_time

    # Always show a summary
    print()
    print("=" * 60)
    print("📊 ANALYSIS SUMMARY")
    print("=" * 60)

    total = len(results)
    injections = sum(1 for r in results if r["is_injection"])
    safe = total - injections

    # Calculate total size
    total_size = sum(r["size"] for r in results)
    avg_size = total_size / total if total > 0 else 0

    # Format sizes
    total_size_str = f"{total_size:,} bytes"
    if total_size > 1024:
        total_size_str = f"{total_size / 1024:.1f} KB"

    avg_size_str = f"{avg_size:.1f} bytes"
    if avg_size > 1024:
        avg_size_str = f"{avg_size / 1024:.1f} KB"

    # Display statistics
    print(f"📁 Directory: {directory_path}")
    print(f"📄 Files analyzed: {total}")
    print(f"📦 Total size: {total_size_str}")
    print(f"📏 Average file size: {avg_size_str}")
    print(f"⏱️  Analysis time: {total_time:.2f} seconds")
    print()

    # Results breakdown
    print("📈 Results:")
    injection_pct = (injections / total * 100) if total > 0 else 0
    safe_pct = (safe / total * 100) if total > 0 else 0

    # Create visual bars
    bar_length = 20
    injection_bar = "█" * int((injections / total) * bar_length) if total > 0 else ""
    safe_bar = "█" * int((safe / total) * bar_length) if total > 0 else ""

    print(f"  🔴 Injections: {injections:3d} ({injection_pct:5.1f}%) {injection_bar}")
    print(f"  🟢 Safe:       {safe:3d} ({safe_pct:5.1f}%) {safe_bar}")
    print()

    # Show top injection candidates if any
    if injections > 0:
        injection_results = [r for r in results if r["is_injection"]]
        injection_results.sort(key=lambda x: x["confidence"], reverse=True)

        print("🔴 Top Injection Candidates:")
        for i, r in enumerate(injection_results[:5], 1):
            # Truncate text for display
            preview = r["text"][:60].replace("\n", " ").replace("\r", "")
            if len(r["text"]) > 60:
                preview += "..."

            print(f"  {i}. {r['file']}")
            print(f"     Confidence: {r['confidence']:.1%}")
            print(f"     Preview: {preview}")
            print()

    # Show top safe files with high confidence
    if safe > 0:
        safe_results = [r for r in results if not r["is_injection"]]
        safe_results.sort(key=lambda x: x["confidence"], reverse=True)

        if len(safe_results) > 0 and safe_results[0]["confidence"] > 0.9:
            print("🟢 Most Confidently Safe:")
            for i, r in enumerate(safe_results[:3], 1):
                if r["confidence"] > 0.9:
                    preview = r["text"][:60].replace("\n", " ").replace("\r", "")
                    if len(r["text"]) > 60:
                        preview += "..."

                    print(f"  {i}. {r['file']}")
                    print(f"     Confidence: {r['confidence']:.1%}")
                    print(f"     Preview: {preview}")
                    print()

    print("=" * 60)
    print("✅ Analysis complete!")


def main():
    """Main function for backward compatibility."""
    import warnings

    warnings.warn(
        "Using detector.py directly is deprecated. Use 'promptscan' CLI instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    legacy_main()


if __name__ == "__main__":
    main()
