#!/usr/bin/env python3
"""
Aggregate all data from different sources into consolidated parquet files.
Sources:
1. prompts.json (original dataset)
2. data/prompts.db (SQLite database)
3. data/external/*.csv (external CSV files)
4. data/processed/*.csv (processed CSV files)
"""

import hashlib
import json
import sqlite3
import warnings
from pathlib import Path
from typing import Dict

import pandas as pd

warnings.filterwarnings("ignore")


def load_json_data(json_path: Path) -> pd.DataFrame:
    """Load data from JSON file."""
    print(f"Loading JSON data from {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    print(f"  Loaded {len(df)} records from JSON")
    return df


def load_sqlite_data(db_path: Path) -> pd.DataFrame:
    """Load data from SQLite database."""
    print(f"Loading SQLite data from {db_path}...")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT id, text, is_injection FROM prompts", conn)
    conn.close()

    # Convert boolean column
    df["is_injection"] = df["is_injection"].astype(bool)
    print(f"  Loaded {len(df)} records from SQLite")
    return df


def load_csv_files(csv_dir: Path, pattern: str = "*.csv") -> pd.DataFrame:
    """Load all CSV files from directory."""
    print(f"Loading CSV files from {csv_dir}...")
    csv_files = list(csv_dir.glob(pattern))

    if not csv_files:
        print(f"  No CSV files found in {csv_dir}")
        return pd.DataFrame()

    all_dfs = []
    for csv_file in csv_files:
        try:
            # Try different encodings
            try:
                df = pd.read_csv(csv_file, encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(csv_file, encoding="latin-1")

            # Standardize column names
            column_map = {}
            for col in df.columns:
                col_lower = col.lower().strip()
                if "prompt" in col_lower or "text" in col_lower:
                    column_map[col] = "text"
                elif (
                    "type" in col_lower
                    or "label" in col_lower
                    or "injection" in col_lower
                ):
                    column_map[col] = "is_injection"
                elif "id" in col_lower:
                    column_map[col] = "id"

            df = df.rename(columns=column_map)

            # Ensure required columns exist
            if "text" not in df.columns:
                print(f"  Warning: 'text' column not found in {csv_file.name}")
                continue

            if "is_injection" not in df.columns:
                print(f"  Warning: 'is_injection' column not found in {csv_file.name}")
                continue

            # Convert is_injection to boolean
            if df["is_injection"].dtype == "object":
                # Try to convert string values to boolean
                df["is_injection"] = (
                    df["is_injection"]
                    .astype(str)
                    .str.lower()
                    .map(
                        {
                            "true": True,
                            "false": False,
                            "1": True,
                            "0": False,
                            "yes": True,
                            "no": False,
                            "injection": True,
                            "benign": False,
                            "malicious": True,
                            "safe": False,
                        }
                    )
                )

            df["is_injection"] = df["is_injection"].astype(bool)

            # Add source column
            df["source"] = csv_file.name

            all_dfs.append(df)
            print(f"  Loaded {len(df)} records from {csv_file.name}")

        except Exception as e:
            print(f"  Error loading {csv_file.name}: {e}")
            continue

    if not all_dfs:
        return pd.DataFrame()

    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"  Total CSV records: {len(combined_df)}")
    return combined_df


def create_text_hash(text: str) -> str:
    """Create hash of text for deduplication."""
    return hashlib.md5(text.strip().lower().encode("utf-8")).hexdigest()


def deduplicate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate records based on text content."""
    print("Deduplicating data...")

    # Create hash for each text
    df["text_hash"] = df["text"].apply(create_text_hash)

    # Find duplicates
    duplicate_mask = df.duplicated(subset=["text_hash"], keep="first")
    duplicates_count = duplicate_mask.sum()

    if duplicates_count > 0:
        print(f"  Found {duplicates_count} duplicate records")

        # For duplicates, prefer injection labels over safe labels
        # (if there's conflict in labeling)
        unique_df = df.sort_values("is_injection", ascending=False).drop_duplicates(
            subset=["text_hash"], keep="first"
        )

        print(f"  After deduplication: {len(unique_df)} unique records")
        return unique_df.drop(columns=["text_hash"])
    else:
        print("  No duplicates found")
        return df.drop(columns=["text_hash"])


def split_data(
    df: pd.DataFrame, train_ratio: float = 0.8, val_ratio: float = 0.1
) -> Dict[str, pd.DataFrame]:
    """Split data into train, validation, and test sets."""
    print(
        f"\nSplitting data (train: {train_ratio}, val: {val_ratio}, test: {1 - train_ratio - val_ratio})..."
    )

    if len(df) == 0:
        return {"train": pd.DataFrame(), "val": pd.DataFrame(), "test": pd.DataFrame()}

    # Shuffle the data
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Calculate split sizes
    n_total = len(df_shuffled)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)

    # Split the data
    train_df = df_shuffled.iloc[:n_train]
    val_df = df_shuffled.iloc[n_train : n_train + n_val]
    test_df = df_shuffled.iloc[n_train + n_val :]

    print(f"  Train: {len(train_df)} samples ({len(train_df) / n_total:.1%})")
    print(f"  Validation: {len(val_df)} samples ({len(val_df) / n_total:.1%})")
    print(f"  Test: {len(test_df)} samples ({len(test_df) / n_total:.1%})")

    # Check class distribution
    print("\n  Class distribution:")
    for name, split_df in [
        ("Train", train_df),
        ("Validation", val_df),
        ("Test", test_df),
    ]:
        if len(split_df) > 0:
            injection_rate = split_df["is_injection"].mean()
            print(f"    {name}: {injection_rate:.2%} injections")

    return {"train": train_df, "val": val_df, "test": test_df}


def save_to_parquet(df: pd.DataFrame, filepath: Path) -> None:
    """Save DataFrame to Parquet format."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(filepath, index=False)
    print(f"Saved {len(df)} records to {filepath}")
    print(f"  File size: {filepath.stat().st_size / 1024 / 1024:.2f} MB")


def main():
    """Main function."""
    print("NOTE: This script creates static split files which are deprecated.")
    print("   Modern training uses merged.parquet with dynamic splits.")
    print(
        "   The generated train/val/test.parquet files are for backward compatibility only."
    )
    print("=" * 60)
    print("Aggregating data from all sources...")
    print("=" * 60)
    print("Aggregating all data sources")
    print("=" * 60)

    # Define paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"

    # Load data from all sources
    all_dataframes = []

    # 1. Load from JSON
    json_path = base_dir / "prompts.json"
    if json_path.exists():
        json_df = load_json_data(json_path)
        json_df["source"] = "prompts.json"
        all_dataframes.append(json_df)

    # 2. Load from SQLite
    db_path = data_dir / "prompts.db"
    if db_path.exists():
        sqlite_df = load_sqlite_data(db_path)
        sqlite_df["source"] = "prompts.db"
        all_dataframes.append(sqlite_df)

    # 3. Load from external CSV files
    external_dir = data_dir / "external"
    if external_dir.exists():
        external_df = load_csv_files(external_dir)
        if not external_df.empty:
            all_dataframes.append(external_df)

    # 4. Load from processed CSV files
    processed_dir = data_dir / "processed"
    if processed_dir.exists():
        processed_df = load_csv_files(processed_dir)
        if not processed_df.empty:
            all_dataframes.append(processed_df)

    if not all_dataframes:
        print("Error: No data sources found!")
        return 1

    # Combine all data
    print(f"\nCombining data from {len(all_dataframes)} sources...")
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"Total records before deduplication: {len(combined_df)}")

    # Standardize columns
    required_columns = ["text", "is_injection"]
    for col in required_columns:
        if col not in combined_df.columns:
            print(f"Error: Required column '{col}' not found!")
            return 1

    # Keep only required columns plus source
    combined_df = combined_df[["text", "is_injection", "source"]]

    # Clean text
    combined_df["text"] = combined_df["text"].astype(str).str.strip()

    # Remove empty texts
    original_count = len(combined_df)
    combined_df = combined_df[combined_df["text"].str.len() > 0]
    print(f"Removed {original_count - len(combined_df)} empty texts")

    # Deduplicate
    combined_df = deduplicate_data(combined_df)

    # Analyze sources
    print("\nData sources analysis:")
    source_counts = combined_df["source"].value_counts()
    for source, count in source_counts.items():
        injection_rate = combined_df[combined_df["source"] == source][
            "is_injection"
        ].mean()
        print(f"  {source}: {count} records ({injection_rate:.2%} injections)")

    # Overall statistics
    print("\nOverall statistics:")
    print(f"  Total unique records: {len(combined_df)}")
    print(f"  Injection rate: {combined_df['is_injection'].mean():.2%}")
    print(f"  Safe prompts: {(combined_df['is_injection'] == False).sum()}")
    print(f"  Injection prompts: {(combined_df['is_injection'] == True).sum()}")

    # Split data
    splits = split_data(combined_df)

    # Save splits
    print("\nSaving data splits...")
    save_to_parquet(splits["train"], data_dir / "train.parquet")
    save_to_parquet(splits["val"], data_dir / "val.parquet")
    save_to_parquet(splits["test"], data_dir / "test.parquet")

    # Also save full dataset for reference
    save_to_parquet(combined_df, data_dir / "prompts_full.parquet")

    # Create backup of original files before removal
    print("\nCreating backups of original files...")
    backup_dir = data_dir / "backup_original"
    backup_dir.mkdir(exist_ok=True)

    files_to_backup = [
        (json_path, backup_dir / "prompts.json"),
        (db_path, backup_dir / "prompts.db"),
        (external_dir, backup_dir / "external"),
        (processed_dir, backup_dir / "processed"),
        (data_dir / "merged.parquet", backup_dir / "merged.parquet"),
        (data_dir / "prompts.db.backup", backup_dir / "prompts.db.backup"),
    ]

    import shutil

    for src, dst in files_to_backup:
        if src.exists():
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
            print(f"  Backed up: {src.name}")

    print("\n" + "=" * 60)
    print("Aggregation complete!")
    print("=" * 60)
    print("\nGenerated files:")
    print(f"  train.parquet: {len(splits['train'])} records")
    print(f"  val.parquet: {len(splits['val'])} records")
    print(f"  test.parquet: {len(splits['test'])} records")
    print(f"  prompts_full.parquet: {len(combined_df)} records (full dataset)")
    print(f"\nOriginal files backed up to: {backup_dir}")

    return 0


if __name__ == "__main__":
    exit(main())
