#!/usr/bin/env python3
"""
Consolidate all parquet files into a single merged.parquet file.
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import uuid
import sys
from typing import List, Dict, Any
import warnings

warnings.filterwarnings("ignore")


def load_and_normalize(filepath: Path) -> pd.DataFrame:
    """Load a parquet file and normalize its schema."""
    print(f"Loading {filepath.name}...")
    df = pd.read_parquet(filepath)

    # Normalize column names and types
    normalized = {}

    # Text column (required)
    if "text" in df.columns:
        normalized["text"] = df["text"].astype(str)
    elif "prompt" in df.columns:
        normalized["text"] = df["prompt"].astype(str)
    else:
        raise ValueError(f"No text column found in {filepath.name}")

    # is_injection column (required)
    if "is_injection" in df.columns:
        normalized["is_injection"] = df["is_injection"].astype(bool)
    elif "label" in df.columns:
        # Convert 'safe'/'injection' labels to boolean
        normalized["is_injection"] = df["label"].apply(
            lambda x: x.lower() == "injection" if isinstance(x, str) else bool(x)
        )
    else:
        # Default to False if no label information
        normalized["is_injection"] = False

    # Source column (optional)
    if "source" in df.columns:
        normalized["source"] = df["source"].fillna("unknown").astype(str)
    else:
        normalized["source"] = "unknown"

    # ID column (optional - will be generated if missing)
    if "id" in df.columns:
        normalized["id"] = df["id"].astype(str)

    # text_length column (optional - will be calculated if missing)
    if "text_length" in df.columns:
        normalized["text_length"] = df["text_length"].astype(int)

    return pd.DataFrame(normalized)


def consolidate_data(data_dir: Path = Path("data")) -> pd.DataFrame:
    """Consolidate all parquet files into a single DataFrame."""
    all_data = []

    # Files to process (excluding the consolidated output and unverified submissions)
    files_to_process = [
        "processed.parquet",   # output of merge_hf_datasets.py
        "prompts_full.parquet",
        "unified_prompts.parquet",
        "train.parquet",
        "val.parquet",
        "test.parquet",
        "train_split.parquet",
        "val_split.parquet",
        "test_split.parquet",
    ]

    # Preserve existing merged.parquet data (may contain user-submitted prompts
    # not present in any of the above files). main() renames merged.parquet to
    # merged_backup.parquet before calling this function, so check both names.
    for merged_name in ("merged_backup.parquet", "merged.parquet"):
        existing_merged = data_dir / merged_name
        if existing_merged.exists():
            try:
                df = load_and_normalize(existing_merged)
                all_data.append(df)
                print(f"  ✓ Loaded {len(df)} rows from {merged_name}")
            except Exception as e:
                print(f"  ✗ Error loading {merged_name}: {e}")
            break

    for filename in files_to_process:
        filepath = data_dir / filename
        if filepath.exists():
            try:
                df = load_and_normalize(filepath)
                all_data.append(df)
                print(f"  ✓ Loaded {len(df)} rows from {filename}")
            except Exception as e:
                print(f"  ✗ Error loading {filename}: {e}")

    # Also include the legacy prompts.parquet if it exists (for backward compatibility)
    prompts_file = data_dir / "prompts.parquet"
    if prompts_file.exists():
        try:
            df = load_and_normalize(prompts_file)
            all_data.append(df)
            print(f"  ✓ Loaded {len(df)} rows from legacy prompts.parquet")
        except Exception as e:
            print(f"  ✗ Error loading legacy prompts.parquet: {e}")

    if not all_data:
        raise ValueError("No data files found to consolidate")

    # Combine all data
    combined = pd.concat(all_data, ignore_index=True)

    # Remove exact duplicates based on text and is_injection
    combined = combined.drop_duplicates(subset=["text", "is_injection"])

    # Generate consistent IDs for all rows
    print("Generating consistent IDs...")
    combined["id"] = [str(uuid.uuid4()) for _ in range(len(combined))]

    # Calculate text_length if not present
    if "text_length" not in combined.columns:
        print("Calculating text lengths...")
        combined["text_length"] = combined["text"].str.len()

    # Ensure consistent column order
    final_columns = ["id", "text", "is_injection", "source", "text_length"]
    for col in final_columns:
        if col not in combined.columns:
            if col == "source":
                combined["source"] = "consolidated"
            elif col == "text_length":
                combined["text_length"] = combined["text"].str.len()

    combined = combined[final_columns]

    return combined


def save_consolidated_data(df: pd.DataFrame, output_path: Path) -> None:
    """Save consolidated data to parquet file."""
    print(f"\nSaving consolidated data to {output_path}...")

    # Convert to pyarrow table for efficient storage
    table = pa.Table.from_pandas(df, preserve_index=False)

    # Write with compression
    pq.write_table(table, output_path, compression="snappy")

    print(f"✓ Saved {len(df)} rows to {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Print statistics
    print("\n📊 Consolidated Data Statistics:")
    print(f"  Total prompts: {len(df)}")
    print(
        f"  Injection prompts: {df['is_injection'].sum()} ({df['is_injection'].mean() * 100:.1f}%)"
    )
    print(
        f"  Safe prompts: {(~df['is_injection']).sum()} ({100 - df['is_injection'].mean() * 100:.1f}%)"
    )
    print(f"  Average text length: {df['text_length'].mean():.0f} characters")
    print(f"  Unique sources: {df['source'].nunique()}")


def backup_original_files(data_dir: Path) -> None:
    """Create backup of original parquet files."""
    backup_dir = data_dir / "backup"
    backup_dir.mkdir(exist_ok=True)

    print(f"\nCreating backup in {backup_dir}...")

    for filepath in data_dir.glob("*.parquet"):
        if (
            filepath.name != "merged.parquet"
            and filepath.name != "unverified_user_submissions.parquet"
        ):
            backup_path = backup_dir / filepath.name
            if backup_path.exists():
                backup_path.unlink()
            filepath.rename(backup_path)
            print(f"  ✓ Backed up {filepath.name}")

    print("✓ Backup complete")


def cleanup_redundant_files(data_dir: Path) -> None:
    """Remove redundant parquet files (keeping only merged.parquet)."""
    print("\nCleaning up redundant files...")

    files_to_keep = {"merged.parquet", "unverified_user_submissions.parquet"}

    for filepath in data_dir.glob("*.parquet"):
        if filepath.name not in files_to_keep:
            try:
                filepath.unlink()
                print(f"  ✓ Removed {filepath.name}")
            except Exception as e:
                print(f"  ✗ Error removing {filepath.name}: {e}")


def main():
    """Main consolidation script."""
    print("=" * 60)
    print("PROMPTSCAN DATA CONSOLIDATION")
    print("=" * 60)

    data_dir = Path("data")
    output_path = data_dir / "merged.parquet"

    # Create backup of current merged.parquet if it exists
    if output_path.exists():
        backup_path = data_dir / "merged_backup.parquet"
        if backup_path.exists():
            backup_path.unlink()
        output_path.rename(backup_path)
        print(f"✓ Backed up existing merged.parquet to merged_backup.parquet")

    try:
        # Consolidate all data
        consolidated_df = consolidate_data(data_dir)

        # Save consolidated data
        save_consolidated_data(consolidated_df, output_path)

        # Create backup of original files
        backup_original_files(data_dir)

        # Cleanup redundant files
        cleanup_redundant_files(data_dir)

        print("\n" + "=" * 60)
        print("✅ DATA CONSOLIDATION COMPLETE")
        print("=" * 60)
        print(f"\nOnly these files remain in {data_dir}:")
        for f in sorted(data_dir.glob("*.parquet")):
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"  • {f.name} ({size_mb:.1f} MB)")

        print(f"\nBackup files are in {data_dir}/backup/")

    except Exception as e:
        print(f"\n❌ Consolidation failed: {e}")
        print("Restoring original merged.parquet...")
        backup_path = data_dir / "merged_backup.parquet"
        if backup_path.exists():
            backup_path.rename(output_path)
            print("✓ Restored original merged.parquet")
        sys.exit(1)


if __name__ == "__main__":
    main()
