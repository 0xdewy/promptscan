#!/usr/bin/env python3
"""
Unify and deduplicate all prompt data files.
Creates a single, clean dataset for training.
"""

import pandas as pd
import hashlib
from pathlib import Path
from typing import List, Dict, Any
import argparse


def load_and_clean_data(filepath: Path) -> pd.DataFrame:
    """Load a parquet file and clean the data."""
    print(f"Loading {filepath.name}...")
    df = pd.read_parquet(filepath)

    # Ensure required columns exist
    if "text" not in df.columns:
        print(f"  Warning: 'text' column missing in {filepath.name}")
        return None

    if "is_injection" not in df.columns:
        print(f"  Warning: 'is_injection' column missing in {filepath.name}")
        return None

    # Clean text: strip whitespace, convert to string
    df["text"] = df["text"].astype(str).str.strip()

    # Remove empty texts
    original_len = len(df)
    df = df[df["text"].str.len() > 0]
    if len(df) < original_len:
        print(f"  Removed {original_len - len(df)} empty texts")

    # Create hash for deduplication
    df["text_hash"] = df["text"].apply(lambda x: hashlib.md5(x.encode()).hexdigest())

    # Add source column if not present
    if "source" not in df.columns:
        df["source"] = filepath.stem

    print(f"  Loaded {len(df):,} rows")
    return df


def unify_datasets(data_dir: Path = Path("data")) -> pd.DataFrame:
    """Unify all parquet files in data directory."""
    print("=" * 60)
    print("UNIFYING AND DEDUPLICATING DATASETS")
    print("=" * 60)

    # Find all parquet files
    parquet_files = list(data_dir.glob("*.parquet"))
    print(f"Found {len(parquet_files)} parquet files")

    # Load all valid datasets
    all_dfs = []
    for filepath in parquet_files:
        if filepath.name == "unified_prompts.parquet":
            continue  # Skip output file if it exists

        df = load_and_clean_data(filepath)
        if df is not None:
            all_dfs.append(df)

    if not all_dfs:
        print("No valid datasets found!")
        return None

    # Combine all datasets
    print(f"\nCombining {len(all_dfs)} datasets...")
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"Total rows before deduplication: {len(combined):,}")

    # Deduplicate by text hash
    print("\nDeduplicating...")
    before_dedup = len(combined)
    combined = combined.drop_duplicates(subset=["text_hash"], keep="first")
    after_dedup = len(combined)
    duplicates_removed = before_dedup - after_dedup

    print(
        f"Removed {duplicates_removed:,} duplicates ({duplicates_removed / before_dedup * 100:.1f}%)"
    )
    print(f"Total rows after deduplication: {after_dedup:,}")

    # Remove temporary hash column
    combined = combined.drop(columns=["text_hash"])

    # Add unique IDs
    combined = combined.reset_index(drop=True)
    combined["id"] = combined.index

    # Reorder columns
    column_order = ["id", "text", "is_injection", "source"]
    extra_cols = [c for c in combined.columns if c not in column_order]
    combined = combined[column_order + extra_cols]

    return combined


def analyze_dataset(df: pd.DataFrame) -> None:
    """Analyze and print dataset statistics."""
    print("\n" + "=" * 60)
    print("DATASET ANALYSIS")
    print("=" * 60)

    print(f"Total samples: {len(df):,}")

    # Label distribution
    injections = df["is_injection"].sum()
    safe = len(df) - injections
    print(f"\nLabel distribution:")
    print(f"  Injections: {injections:,} ({injections / len(df) * 100:.1f}%)")
    print(f"  Safe: {safe:,} ({safe / len(df) * 100:.1f}%)")

    # Source distribution
    print(f"\nSource distribution:")
    source_counts = df["source"].value_counts()
    for source, count in source_counts.items():
        print(f"  {source}: {count:,} ({count / len(df) * 100:.1f}%)")

    # Text length statistics
    df["text_length"] = df["text"].str.len()
    print(f"\nText length statistics:")
    print(f"  Min: {df['text_length'].min():.0f} chars")
    print(f"  Max: {df['text_length'].max():.0f} chars")
    print(f"  Mean: {df['text_length'].mean():.1f} chars")
    print(f"  Median: {df['text_length'].median():.1f} chars")

    # Sample preview
    print(f"\nSample preview (first 5 rows):")
    for i, row in df.head().iterrows():
        label = "INJECTION" if row["is_injection"] else "SAFE"
        preview = row["text"][:80] + "..." if len(row["text"]) > 80 else row["text"]
        print(f"  [{label}] {preview}")


def create_train_val_test_splits(
    df: pd.DataFrame, output_dir: Path = Path("data")
) -> None:
    """Create train/val/test splits from unified dataset."""
    print("\n" + "=" * 60)
    print("CREATING TRAINING SPLITS")
    print("=" * 60)

    # Use existing splits if available in source
    if (
        "train" in df["source"].values
        and "val" in df["source"].values
        and "test" in df["source"].values
    ):
        print("Using existing splits from source...")
        train_df = df[df["source"] == "train"].copy()
        val_df = df[df["source"] == "val"].copy()
        test_df = df[df["source"] == "test"].copy()

        # Remove source column to match expected format
        train_df = train_df.drop(columns=["source"])
        val_df = val_df.drop(columns=["source"])
        test_df = test_df.drop(columns=["source"])
    else:
        # Create new splits (80/10/10)
        print("Creating new splits (80/10/10)...")
        from sklearn.model_selection import train_test_split

        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            df, test_size=0.1, random_state=42, stratify=df["is_injection"]
        )

        # Second split: train vs val
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=0.1111,
            random_state=42,
            stratify=train_val_df["is_injection"],
        )  # 0.1111 = 0.1 / 0.9

        # Remove source column
        train_df = train_df.drop(columns=["source"])
        val_df = val_df.drop(columns=["source"])
        test_df = test_df.drop(columns=["source"])

    print(f"Training set: {len(train_df):,} samples")
    print(f"Validation set: {len(val_df):,} samples")
    print(f"Test set: {len(test_df):,} samples")

    # Save splits
    train_path = output_dir / "train_split.parquet"
    val_path = output_dir / "val_split.parquet"
    test_path = output_dir / "test_split.parquet"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print(f"\nSaved splits to:")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print(f"  {test_path}")

    return train_df, val_df, test_df


def main():
    parser = argparse.ArgumentParser(
        description="Unify and deduplicate prompt datasets"
    )
    parser.add_argument(
        "--data-dir", default="data", help="Directory containing parquet files"
    )
    parser.add_argument(
        "--output", default="data/unified_prompts.parquet", help="Output file path"
    )
    parser.add_argument(
        "--create-splits", action="store_true", help="Create train/val/test splits"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_path = Path(args.output)

    # Check if output exists
    if output_path.exists() and not args.overwrite:
        print(f"Output file {output_path} already exists. Use --overwrite to replace.")
        return

    # Unify datasets
    unified_df = unify_datasets(data_dir)
    if unified_df is None:
        return

    # Analyze dataset
    analyze_dataset(unified_df)

    # Save unified dataset
    print(f"\nSaving unified dataset to {output_path}...")
    unified_df.to_parquet(output_path, index=False)
    print(f"Saved {len(unified_df):,} samples")

    # Create splits if requested
    if args.create_splits:
        create_train_val_test_splits(unified_df, data_dir)

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
