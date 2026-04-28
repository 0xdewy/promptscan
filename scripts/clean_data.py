#!/usr/bin/env python3
"""
Data cleaning script for promptscan merged.parquet dataset.

Applies fixes to produce a clean dataset:
- Computes text_length for all rows
- Removes suspicious/internal sources
- Deduplicates by text content
- Drops unused columns
- Generates samples for manual label review

Usage:
    python scripts/clean_data.py
"""

import random
from pathlib import Path

import pandas as pd


SUSPICIOUS_PATTERNS = [
    "prompts.db",
    "prompts.json",
    "../",
    "promptscan-website",
    ".sh",
    ".py",
    ".js",
    ".html",
    ".csv",
]


def load_data(path: str = "data/merged.parquet") -> pd.DataFrame:
    """Load the merged parquet dataset."""
    print(f"Loading data from {path}...")
    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} rows with {len(df.columns)} columns")
    return df


def compute_text_length(df: pd.DataFrame) -> pd.DataFrame:
    """Compute text_length for all rows."""
    print("\n=== COMPUTING TEXT LENGTH ===")
    before_nulls = df["text_length"].isnull().sum()
    df["text_length"] = df["text"].str.len()
    after_nulls = df["text_length"].isnull().sum()
    print(f"  Computed text_length for {before_nulls:,} rows (was all null)")
    return df


def remove_suspicious_sources(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows from suspicious/internal sources."""
    import re
    print("\n=== REMOVING SUSPICIOUS SOURCES ===")
    pattern = "|".join(re.escape(p) for p in SUSPICIOUS_PATTERNS)
    suspicious_mask = df["source"].str.contains(pattern, na=False)
    suspicious_count = suspicious_mask.sum()
    suspicious_sources = df[suspicious_mask]["source"].unique().tolist()

    print(
        f"  Found {suspicious_count:,} rows from {len(suspicious_sources)} suspicious sources"
    )
    print(f"  Sources being removed:")
    for src in suspicious_sources[:10]:
        count = len(df[df["source"] == src])
        print(f"    - {src}: {count} rows")
    if len(suspicious_sources) > 10:
        print(f"    ... and {len(suspicious_sources) - 10} more")

    df = df[~suspicious_mask].copy()
    print(f"  Remaining rows: {len(df):,}")
    return df


def filter_short_texts(df: pd.DataFrame, min_length: int = 3) -> pd.DataFrame:
    """Remove texts that are too short to carry meaningful signal."""
    print(f"\n=== FILTERING SHORT TEXTS (min {min_length} chars) ===")
    before = len(df)
    df = df[df["text"].str.len() >= min_length].copy()
    removed = before - len(df)
    print(f"  Removed {removed:,} texts shorter than {min_length} chars")
    print(f"  Remaining rows: {len(df):,}")
    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate texts, keeping first occurrence."""
    print("\n=== DEDUPLICATING ===")
    before = len(df)
    df = df.drop_duplicates(subset=["text"], keep="first")
    removed = before - len(df)
    print(f"  Removed {removed:,} duplicate texts")
    print(f"  Remaining rows: {len(df):,}")
    return df


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are >99% null or unused metadata columns."""
    print("\n=== DROPPING UNUSED COLUMNS ===")
    unused_cols = [
        "review_date",
        "original_prediction",
        "original_user_label",
        "original_confidence",
        "original_source",
        "original_timestamp",
    ]

    cols_to_drop = [c for c in unused_cols if c in df.columns]
    null_cols = [c for c in cols_to_drop if df[c].isnull().mean() > 0.99]

    print(f"  Dropping {len(null_cols)} columns (>99% null):")
    for col in null_cols:
        null_pct = df[col].isnull().mean() * 100
        print(f"    - {col} ({null_pct:.2f}% null)")

    df = df.drop(columns=null_cols)
    print(f"  Remaining columns: {df.columns.tolist()}")
    return df


def generate_review_samples(df: pd.DataFrame, n_per_class: int = 100) -> pd.DataFrame:
    """Generate stratified samples for manual label review."""
    print(f"\n=== GENERATING REVIEW SAMPLES ({n_per_class} per class) ===")

    samples = []
    random.seed(42)

    for label in [True, False]:
        label_df = df[df["is_injection"] == label]
        sample = label_df.sample(n=min(n_per_class, len(label_df)), random_state=42)
        for _, row in sample.iterrows():
            samples.append(
                {
                    "id": row["id"],
                    "text": row["text"],
                    "is_injection": row["is_injection"],
                    "source": row["source"],
                    "text_length": row["text_length"],
                    "review_status": "pending",
                    "correct_label": "",
                    "notes": "",
                }
            )

    samples_df = pd.DataFrame(samples)
    output_path = "data/validation_samples.csv"
    samples_df.to_csv(output_path, index=False)
    print(f"  Saved {len(samples_df)} samples to {output_path}")
    print(f"  Please review and fill in 'correct_label' and 'notes' columns")

    return samples_df


def reset_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """Reset id column to be sequential after cleaning."""
    print("\n=== RESETTING ID COLUMN ===")
    df["id"] = [f"row-{i}" for i in range(len(df))]
    print(f"  IDs range from row-0 to row-{len(df) - 1}")
    return df


def save_clean_data(
    df: pd.DataFrame, output_path: str = "data/merged_clean.parquet"
) -> None:
    """Save the cleaned dataset."""
    print(f"\n=== SAVING CLEANED DATA ===")
    df.to_parquet(output_path, index=False)
    print(f"  Saved {len(df):,} rows to {output_path}")


def print_summary(df: pd.DataFrame, original_rows: int) -> None:
    """Print cleaning summary."""
    print("\n" + "=" * 60)
    print("CLEANING SUMMARY")
    print("=" * 60)
    print(f"  Original rows:     {original_rows:,}")
    print(f"  Cleaned rows:       {len(df):,}")
    print(f"  Rows removed:       {original_rows - len(df):,}")
    print(
        f"  Removal rate:       {(original_rows - len(df)) / original_rows * 100:.2f}%"
    )
    print(f"\n  Final columns:      {df.columns.tolist()}")
    print(f"\n  Class distribution:")
    print(
        f"    is_injection=True:  {df['is_injection'].sum():,} ({df['is_injection'].mean() * 100:.2f}%)"
    )
    print(
        f"    is_injection=False: {(~df['is_injection']).sum():,} ({(1 - df['is_injection'].mean()) * 100:.2f}%)"
    )
    print(f"\n  Sources remaining:   {df['source'].nunique():,}")
    print(f"\n  Null counts:")
    for col in df.columns:
        nulls = df[col].isnull().sum()
        if nulls > 0:
            print(f"    {col}: {nulls:,}")


def main():
    """Run full cleaning pipeline."""
    print("=" * 60)
    print("PROMPTSCAN DATA CLEANING")
    print("=" * 60)

    df = load_data()
    original_rows = len(df)

    df = compute_text_length(df)
    df = remove_suspicious_sources(df)
    df = filter_short_texts(df)
    df = deduplicate(df)
    df = drop_unused_columns(df)
    df = reset_id_column(df)

    generate_review_samples(df, n_per_class=100)
    save_clean_data(df)
    print_summary(df, original_rows)

    print("\n" + "=" * 60)
    print("CLEANING COMPLETE")
    print("=" * 60)
    print("\nOutputs:")
    print("  - data/merged_clean.parquet  (cleaned dataset)")
    print("  - data/validation_samples.csv  (200 samples for manual label review)")
    print("\nNext steps:")
    print("  1. Review data/validation_samples.csv")
    print("  2. Update labels based on review findings")
    print(
        "  3. Re-run validation: python scripts/validate_data.py --input data/merged_clean.parquet"
    )


if __name__ == "__main__":
    main()
