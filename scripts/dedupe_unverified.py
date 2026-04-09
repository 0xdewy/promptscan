#!/usr/bin/env python3
"""
Deduplicate unverified_user_submissions.parquet file.
Removes duplicate entries with the same text, predicted_label, and user_label.
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from promptscan.feedback_store import ParquetFeedbackStore


def deduplicate_feedback_file(filepath: str) -> None:
    """Remove duplicate entries from feedback file."""
    print(f"Loading {filepath}...")
    store = ParquetFeedbackStore(filepath)
    df = store.export_to_dataframe()

    print(f"Original entries: {len(df)}")

    # Check for duplicates
    # We consider entries duplicates if they have the same:
    # - text (normalized: stripped, lowercased)
    # - predicted_label
    # - user_label
    df["text_normalized"] = df["text"].str.strip().str.lower()

    # Create a composite key for duplicate detection
    df["composite_key"] = (
        df["text_normalized"] + "|" + df["predicted_label"] + "|" + df["user_label"]
    )

    # Find duplicates
    duplicate_mask = df.duplicated(subset=["composite_key"], keep="first")
    duplicate_count = duplicate_mask.sum()

    print(f"Duplicate entries found: {duplicate_count}")
    print(f"Unique entries: {len(df) - duplicate_count}")

    if duplicate_count == 0:
        print("No duplicates found. File is clean.")
        return

    # Show duplicate distribution
    print("\nDuplicate distribution:")
    dup_counts = df["composite_key"].value_counts()
    for key, count in dup_counts[dup_counts > 1].head(10).items():
        text_preview = key.split("|")[0][:80]
        print(f"  {count} duplicates: '{text_preview}...'")

    # Keep only first occurrence of each duplicate
    df_deduped = df[~duplicate_mask].copy()

    # Remove temporary columns
    df_deduped = df_deduped.drop(columns=["text_normalized", "composite_key"])

    # Reset IDs to be sequential
    df_deduped = df_deduped.reset_index(drop=True)
    df_deduped["id"] = df_deduped.index + 1

    print(f"\nAfter deduplication: {len(df_deduped)} entries")

    # Save back
    store._data = df_deduped
    store._save_data()

    print(f"✅ Deduplicated file saved to {filepath}")

    # Verify
    store2 = ParquetFeedbackStore(filepath)
    df_verify = store2.export_to_dataframe()
    print(f"Verification: {len(df_verify)} entries in file")

    # Check for remaining duplicates
    df_verify["text_normalized"] = df_verify["text"].str.strip().str.lower()
    df_verify["composite_key"] = (
        df_verify["text_normalized"]
        + "|"
        + df_verify["predicted_label"]
        + "|"
        + df_verify["user_label"]
    )
    remaining_dups = df_verify.duplicated(subset=["composite_key"], keep="first").sum()

    if remaining_dups == 0:
        print("✅ Verification passed: No duplicates remaining")
    else:
        print(f"⚠️  Verification warning: {remaining_dups} duplicates still found")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Deduplicate unverified submissions file"
    )
    parser.add_argument(
        "--file",
        default="website/api/data/unverified_user_submissions.parquet",
        help="Path to unverified submissions parquet file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )

    args = parser.parse_args()

    filepath = args.file
    if not Path(filepath).exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)

    if args.dry_run:
        print("DRY RUN - No changes will be made")
        # Load and analyze without saving
        store = ParquetFeedbackStore(filepath)
        df = store.export_to_dataframe()

        print(f"Entries: {len(df)}")

        df["text_normalized"] = df["text"].str.strip().str.lower()
        df["composite_key"] = (
            df["text_normalized"] + "|" + df["predicted_label"] + "|" + df["user_label"]
        )

        duplicate_mask = df.duplicated(subset=["composite_key"], keep="first")
        duplicate_count = duplicate_mask.sum()

        print(f"Duplicates that would be removed: {duplicate_count}")
        print(f"Unique entries that would remain: {len(df) - duplicate_count}")

        if duplicate_count > 0:
            print("\nTop duplicate groups:")
            dup_counts = df["composite_key"].value_counts()
            for key, count in dup_counts[dup_counts > 1].head(5).items():
                parts = key.split("|")
                text_preview = parts[0][:80]
                pred_label = parts[1]
                user_label = parts[2]
                print(f"  {count} duplicates:")
                print(f"    Text: '{text_preview}...'")
                print(f"    Predicted: {pred_label}, User: {user_label}")
                print()
    else:
        deduplicate_feedback_file(filepath)


if __name__ == "__main__":
    main()
