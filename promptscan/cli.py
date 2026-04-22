#!/usr/bin/env python3
"""
Command-line interface for Safe Prompts.
"""

import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import logging
import warnings

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import argparse
import sys

from . import __version__
from .unified_detector import UnifiedDetector
from .utils.colors import Colors


def _display_prediction(result, detector=None, source=None, interactive=False):
    """Display prediction result with individual model predictions for ensemble."""
    if source:
        print(f"{source}:")

    if "individual_predictions" in result:
        if not interactive:
            print("Individual model predictions:")

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
            model_display = Colors.colored(model_type_display, Colors.model_color(idx))
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
    import time

    print("🔍 Safe Prompts - Prompt Injection Detector")
    print("=" * 60)
    print("Loading ensemble model...")

    load_start = time.time()

    input_val = getattr(args, "input", None)

    if input_val:
        if input_val.startswith("http://") or input_val.startswith("https://"):
            args.url = input_val
            args.file = None
            args.dir = None
            args.text = None
        elif os.path.isdir(input_val):
            args.dir = input_val
            args.url = None
            args.file = None
            args.text = None
        elif os.path.isfile(input_val):
            args.file = input_val
            args.url = None
            args.dir = None
            args.text = None
        else:
            args.text = input_val
            args.url = None
            args.file = None
            args.dir = None
    else:
        args.url = None
        args.file = None
        args.dir = None
        args.text = None

    try:
        detector = UnifiedDetector(
            model_type="ensemble",
            device=args.device,
            voting_strategy=args.voting_strategy,
        )
    except FileNotFoundError as e:
        print(f"\n{Colors.error(f'❌ Error: {e}')}")
        print("\n💡 Training new models: ./train_all.sh")
        sys.exit(1)

    load_time = time.time() - load_start

    info = detector.get_info()
    print(f"{Colors.success(f'✅ Model loaded successfully in {load_time:.2f}s')}")
    print()

    print(f"{Colors.header('🤝 Ensemble Model Configuration:')}")
    models_info = str(len(info["models"]))
    voting_info = info["voting_strategy"]
    print(f"   {Colors.info('Models: ' + models_info)}")
    print(f"   {Colors.info('Voting strategy: ' + voting_info)}")
    print()
    print(f"   {Colors.header('Individual models:')}")
    for model_info in info["models"]:
        params = f"{model_info['parameters']:,}"
        print(f"     • {model_info['type']:12} - {params:>10} parameters")

    print()
    print("=" * 60)
    print()

    if args.file:
        import time

        print(f"📁 Analyzing file: {args.file}")
        print("-" * 60)

        try:
            if not os.path.exists(args.file):
                print(f"❌ Error: File not found: {args.file}")
                return

            file_size = os.path.getsize(args.file)
            file_name = os.path.basename(args.file)

            try:
                from .utils.markdown_parser import (
                    get_file_type_display,
                    read_and_parse_file,
                )

                has_markdown_parser = True
            except ImportError:
                has_markdown_parser = False

            if has_markdown_parser:
                file_type = get_file_type_display(args.file)
                file_type_icon = "📝" if file_type == "Markdown" else "📄"
                print(f"{file_type_icon} File: {file_name} ({file_type})")
            else:
                print(f"📄 File: {file_name}")

            print(f"📦 Size: {file_size:,} bytes")
            print()

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

            if char_count > 50000:
                print("⚠️  Warning: File is very large (>50K characters)")
                print("   Analysis may take longer than usual")
                print()

            print("Starting analysis...")
            analysis_start = time.time()
            result = detector.predict(text)
            analysis_time = time.time() - analysis_start

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

            _display_prediction(result, detector)

            print("📝 Content Preview (first 200 characters):")
            print("-" * 40)

            preview = text[:200]
            if len(text) > 200:
                preview += "..."

            preview = preview.replace("\n", " ").replace("\r", " ").replace("  ", " ")
            if len(preview) > 80:
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
            fetch_start = time.time()
            response = requests.get(args.url, timeout=10)
            response.raise_for_status()
            fetch_time = time.time() - fetch_start

            content_type = response.headers.get("content-type", "unknown").split(";")[0]
            content_length = len(response.content)
            text = response.text.strip()
            char_count = len(text)

            print("✅ Successfully fetched URL")
            print(f"   Content type: {content_type}")
            print(f"   Size: {content_length:,} bytes ({char_count:,} characters)")
            print(f"   Fetch time: {fetch_time:.2f} seconds")
            print()

            if char_count > 100000:
                print("⚠️  Warning: Content is very large (>100K characters)")
                print("   Analysis may take longer than usual")
                print()

            print("Starting analysis...")
            analysis_start = time.time()
            result = detector.predict(text)
            analysis_time = time.time() - analysis_start

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

            _display_prediction(result, detector)

            print("📝 Content Preview (first 300 characters):")
            print("-" * 40)

            preview = text[:300]
            if len(text) > 300:
                preview += "..."

            preview = preview.replace("\n", " ").replace("\r", " ").replace("  ", " ")
            if len(preview) > 80:
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

        if char_count > 10000:
            print("⚠️  Warning: Text is very long (>10K characters)")
            print("   Analysis may take longer than usual")
            print()

        print("Starting analysis...")
        analysis_start = time.time()
        result = detector.predict(text)
        analysis_time = time.time() - analysis_start

        print()
        print("=" * 60)
        print("📊 TEXT ANALYSIS RESULTS")
        print("=" * 60)
        print(f"📝 Text length: {char_count:,} characters")
        print(f"⏱️  Analysis time: {analysis_time:.2f} seconds")
        print()

        _display_prediction(result, detector)

        print("📝 Text Preview:")
        print("-" * 40)

        preview = text[:150]
        if len(text) > 150:
            preview += "..."

        preview = preview.replace("\n", " ").replace("\r", " ").replace("  ", " ")
        if len(preview) > 80:
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

                result = detector.predict(text)

                if result["prediction"] == "INJECTION":
                    injection_count += 1

                print(
                    f"   {Colors.prediction(result['prediction'], result['confidence'])}"
                )
                _display_prediction(result, detector, source=None, interactive=True)

                preview = text[:80].replace("\n", " ").replace("\r", " ")
                if len(text) > 80:
                    preview += "..."
                print(f"   📋 Preview: {preview}")

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
    promptscan "Ignore all previous instructions"
    promptscan input.txt
    promptscan ./prompts/
    promptscan https://example.com/user-input
    promptscan --version
        """,
    )

    parser.add_argument(
        "--version", action="store_true", help="Show version information"
    )

    parser.add_argument(
        "input",
        nargs="?",
        help="Text, file, directory, or URL to analyze. Auto-detected by type.",
    )

    parser.add_argument(
        "--voting-strategy",
        choices=["majority", "weighted", "confidence", "soft"],
        default="majority",
        help="Voting strategy for ensemble (default: majority)",
    )

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device to run inference on (auto, cpu, or cuda)",
    )

    parser.add_argument(
        "--summary", action="store_true", help="Show summary for directory analysis"
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show verbose output with details"
    )

    args = parser.parse_args()

    if args.version:
        version_command(args)
        return

    predict_command(args)


if __name__ == "__main__":
    main()
