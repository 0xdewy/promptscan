"""Prompt injection detector — legacy SimplePromptDetector and directory analysis."""

import os
import pickle
from typing import Any, Dict, List

import torch

from . import get_model_path
from .models.cnn_model import SimpleCNN
from .utils.colors import Colors
from .utils.device import get_device


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
            model_path = str(get_model_path("best_model"))
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


def analyze_directory(detector, directory_path, show_summary=False, verbose=False):
    """Analyze all text files (.txt, .md, .markdown) in directory with beautiful output."""
    import glob
    import time

    # Try to import markdown parser
    try:
        from .utils.markdown_parser import get_file_type_display, read_and_parse_file

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
            status = Colors.colored("INJECTION", Colors.RED)
        else:
            icon = "🟢"
            status = Colors.colored("SAFE", Colors.GREEN)

        # Progress indicator
        progress = f"[{i}/{len(text_files)}]"

        # File size indicator
        size_str = f"{file_size:,}B"
        if file_size > 1024:
            size_str = f"{file_size / 1024:.1f}KB"

        # Color confidence based on value
        conf_color = Colors.confidence_color(result["confidence"])
        if Colors.supports_color():
            confidence = f"{conf_color}{result['confidence']:.1%}{Colors.RESET}"
        else:
            confidence = f"{result['confidence']:.1%}"

        print(
            f"{progress} {icon} {file_display:28} {status:10} ({confidence}) {size_str:>8}"
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
