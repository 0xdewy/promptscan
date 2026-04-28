#!/usr/bin/env python3
"""Insert prompts into the database."""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from promptscan.parquet_store import ParquetDataStore
from promptscan.batch_importer import BatchImporter


def _parse_size(size_str: str) -> int:
    """Parse size string like '1MB', '500KB', '1000000' to bytes."""
    size_str = size_str.strip().upper()
    if size_str.endswith("KB"):
        return int(float(size_str[:-2]) * 1024)
    elif size_str.endswith("MB"):
        return int(float(size_str[:-2]) * 1024 * 1024)
    elif size_str.endswith("GB"):
        return int(float(size_str[:-2]) * 1024 * 1024 * 1024)
    else:
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


def import_parquet_file(args, store):
    """Handle importing from a parquet file into the data store."""
    print("📥 Parquet Import Mode")
    print("=" * 60)

    source_path = Path(args.import_from)
    if not source_path.exists():
        print(f"\n❌ Error: Source parquet file not found: {source_path}")
        return

    label = None
    if args.label:
        label = args.label == "injection"
        print(f"🏷️  Label: {args.label}")
    else:
        print("\n❌ Error: --label is required for parquet imports.")
        print("   Use --label safe or --label injection")
        return

    print(f"📄 Source: {source_path}")
    print(f"💾 Target: {store.parquet_path}")
    print(f"🏷️  All prompts will be labeled as: {args.label}")
    print()

    df = pd.read_parquet(source_path)
    print(f"📊 Found {len(df)} prompts in source file")
    print(f"   Columns: {df.columns.tolist()}")

    if "text" not in df.columns:
        print("\n❌ Error: Source parquet must have a 'text' column")
        return

    if "is_injection" not in df.columns:
        print("\n⚠️  Warning: Source parquet has no 'is_injection' column")
        print(f"   Using label: {args.label} for all prompts")
        df["is_injection"] = label
    else:
        df["is_injection"] = df["is_injection"].fillna(label)

    prompts = df[["text", "is_injection"]].to_dict("records")
    added_ids, skipped = store.add_prompts_batch(prompts)

    print(f"\n✅ Import complete:")
    print(f"   Added: {len(added_ids)} prompts")
    print(f"   Duplicates skipped: {skipped}")

    db_stats = store.get_statistics()
    print(f"\n💾 Database now contains {db_stats['total']} prompts:")
    print(
        f"  🔴 {db_stats['injections']} injection prompts ({db_stats['injection_percentage']:.1f}%)"
    )
    print(f"  🟢 {db_stats['safe']} safe prompts ({db_stats['safe_percentage']:.1f}%)")


def batch_insert(args, store):
    """Handle batch insert from GitHub, directory, or file list."""
    print("🔧 Safe Prompts - Batch Insertion Mode")
    print("=" * 60)

    extensions = None
    if args.extensions:
        extensions = [ext.strip() for ext in args.extensions.split(",")]
        extensions = [ext if ext.startswith(".") else f".{ext}" for ext in extensions]
        print(f"📁 Extensions: {', '.join(extensions)}")

    max_size = None
    if args.max_size:
        max_size = _parse_size(args.max_size)
        print(f"📏 Max size: {_format_size(max_size)}")

    label = None
    if args.label:
        label = args.label == "injection"
        print(f"🏷️  Label: {args.label}")

    if (
        args.github or args.dir or (args.file and len(args.file) > 1)
    ) and not args.label:
        print(
            "\n❌ Error: --label is required for batch imports from GitHub, directories, or multiple files."
        )
        print("   Use --label safe or --label injection")
        return

    importer = BatchImporter(
        store, verbose=args.verbose, github_token=args.github_token
    )

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

    if stats:
        print("\n" + "=" * 60)
        print("📊 IMPORT COMPLETE")
        print("=" * 60)
        print(str(stats))

        if not args.dry_run:
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


def interactive_insert(args, store):
    """Handle interactive insert mode."""
    print("💬 Safe Prompts - Interactive Insertion Mode")
    print("=" * 50)
    print("Enter prompts to add to the database.")
    print("Press Ctrl+D (Unix) or Ctrl+Z (Windows) to exit.")
    print()

    try:
        while True:
            text = input("Enter prompt text: ").strip()
            if not text:
                print("Prompt text cannot be empty. Please try again.")
                continue

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

            prompt_id = store.add_prompt(text, is_injection)
            if prompt_id is None:
                label_type = "injection" if is_injection else "safe"
                print(f"⚠️  Prompt already exists in database as {label_type} prompt")
                print()
                continue

            label_type = "injection" if is_injection else "safe"
            print(f"✓ Added prompt #{prompt_id} as {label_type} prompt")
            print()

            while True:
                continue_input = input("Add another prompt? (y/n): ").strip().lower()
                if continue_input in ["y", "yes"]:
                    print()
                    break
                elif continue_input in ["n", "no"]:
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
        stats = store.get_statistics()
        print(f"\n\nDatabase now contains {stats['total']} prompts:")
        print(
            f"  🔴 {stats['injections']} injection prompts ({stats['injection_percentage']:.1f}%)"
        )
        print(f"  🟢 {stats['safe']} safe prompts ({stats['safe_percentage']:.1f}%)")
        print("\n👋 Goodbye!")


def text_insert(args, store):
    """Insert a single raw text prompt directly."""
    is_injection = args.label == "injection"
    prompt_id = store.add_prompt(args.text, is_injection, source="cli_direct_insert")
    label_str = "injection" if is_injection else "safe"
    if prompt_id is None:
        print(f"⚠️  Duplicate — already exists as {label_str} prompt")
    else:
        print(f"✅ Added as {label_str} prompt (id: {prompt_id})")
        stats = store.get_statistics()
        print(
            f"   Database: {stats['total']} total ({stats['injections']} inj / {stats['safe']} safe)"
        )


def main():
    parser = argparse.ArgumentParser(description="Insert new prompts into the database")
    parser.add_argument(
        "--parquet", default="data/merged.parquet", help="Target parquet file path"
    )
    parser.add_argument(
        "--import-from",
        help="Import prompts FROM a parquet file into the target database",
    )
    parser.add_argument(
        "--text",
        "-t",
        help="Raw text to insert as a single prompt",
    )

    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--github", "-g", help="GitHub repository URL")
    source_group.add_argument("--dir", "-d", help="Local directory path")
    source_group.add_argument("--file", "-f", action="append", help="File path(s)")

    parser.add_argument(
        "--extensions",
        help="Comma-separated file extensions to include (e.g., .md,.txt,.py)",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        help="Path patterns to exclude (can be used multiple times)",
    )
    parser.add_argument(
        "--max-size", help="Maximum file size (e.g., 1MB, 500KB, 1000000)"
    )
    parser.add_argument(
        "--batch",
        "-b",
        action="store_true",
        help="Batch mode (no interactive confirmations)",
    )
    parser.add_argument(
        "--label",
        "-l",
        choices=["safe", "injection"],
        help="Label for all items in batch mode",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview without importing"
    )
    parser.add_argument(
        "--output", "-o", help="Output parquet file (default: use --parquet)"
    )
    parser.add_argument(
        "--github-token", help="GitHub personal access token (for higher rate limits)"
    )
    parser.add_argument("--branch", default="main", help="Git branch (default: main)")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show verbose output"
    )

    args = parser.parse_args()

    parquet_file = args.output if args.output else args.parquet
    store = ParquetDataStore(parquet_file)

    if args.text:
        if not args.label:
            print(
                "❌ Error: --label is required with --text. Use --label safe or --label injection"
            )
            sys.exit(1)
        text_insert(args, store)
    elif args.import_from:
        import_parquet_file(args, store)
    elif args.github or args.dir or (args.file and len(args.file) > 0):
        batch_insert(args, store)
    elif args.label and Path(args.parquet).exists():
        args.import_from = args.parquet
        parquet_file = args.output if args.output else "data/merged.parquet"
        store = ParquetDataStore(parquet_file)
        import_parquet_file(args, store)
    else:
        interactive_insert(args, store)


if __name__ == "__main__":
    main()
