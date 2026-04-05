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
    """Main class for prompt injection detection."""

    def __init__(self, model_path=None, device="cpu"):
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
    """Train the model from parquet files."""
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
        train_path, val_path, test_path = get_default_data_paths()
        train_model(
            train_path=str(train_path),
            val_path=str(val_path),
            test_path=str(test_path),
            model_path=args.model,
        )
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
        analyze_directory(detector, args.dir, args.summary)

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


def analyze_directory(detector, directory_path, show_summary=False):
    """Analyze all .txt files in directory."""
    import glob

    txt_files = glob.glob(os.path.join(directory_path, "*.txt"))
    if not txt_files:
        print(f"No .txt files found in {directory_path}")
        return

    results = []
    for file_path in txt_files:
        with open(file_path, "r") as f:
            text = f.read().strip()
        result = detector.predict(text)
        results.append(
            {
                "file": os.path.basename(file_path),
                "text": text[:50] + "..." if len(text) > 50 else text,
                "prediction": result["prediction"],
                "confidence": result["confidence"],
            }
        )

        print(f"{file_path}: {result['prediction']} ({result['confidence']:.2%})")

    if show_summary:
        print("\n=== SUMMARY ===")
        total = len(results)
        injections = sum(1 for r in results if r["prediction"] == "INJECTION")
        safe = total - injections

        print(f"Total texts analyzed: {total}")
        print(f"Injections detected: {injections} ({injections / total * 100:.1f}%)")
        print(f"Safe texts: {safe} ({safe / total * 100:.1f}%)")

        if injections > 0:
            print("\nTop injection candidates:")
            injection_results = [r for r in results if r["prediction"] == "INJECTION"]
            injection_results.sort(key=lambda x: x["confidence"], reverse=True)

            for i, r in enumerate(injection_results[:5], 1):
                print(f"{i}. {r['text']}")
                print(f"   Confidence: {r['confidence']:.1%}, Source: {r['file']}")


def main():
    """Main function for backward compatibility."""
    import warnings

    warnings.warn(
        "Using detector.py directly is deprecated. Use 'prompt-detective' CLI instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    legacy_main()


if __name__ == "__main__":
    main()
