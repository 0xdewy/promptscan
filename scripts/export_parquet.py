#!/usr/bin/env python3
"""
Export prompt injection data from parquet files to various formats.
"""

import argparse
import csv
import json
from pathlib import Path

import pandas as pd


def load_from_parquet(parquet_path: str = "data/merged.parquet") -> pd.DataFrame:
    """Load data from parquet file."""
    # First try the specified path
    path = Path(parquet_path)
    if path.exists():
        df = pd.read_parquet(path)
        print(f"Loaded {len(df)} records from {path}")
        return df

    # Try to find in package data
    try:
        import importlib.resources

        with importlib.resources.path("promptscan", "data") as data_dir:
            package_path = data_dir / "merged.parquet"
            if package_path.exists():
                df = pd.read_parquet(package_path)
                print(f"Loaded {len(df)} records from package data")
                return df
    except (ImportError, FileNotFoundError):
        pass

    # File not found
    raise FileNotFoundError(f"Parquet file not found: {parquet_path}")


def export_to_json(df: pd.DataFrame, output_path: str = "prompts.json"):
    """Export data to JSON format."""
    data = df.to_dict("records")

    # Convert to simpler format (remove id, convert bool to int for compatibility)
    simple_data = []
    for item in data:
        simple_data.append(
            {
                "text": item["text"],
                "is_injection": bool(item["is_injection"]),  # Ensure boolean
            }
        )

    with open(output_path, "w") as f:
        json.dump(simple_data, f, indent=2)

    print(f"Exported {len(simple_data)} prompts to {output_path}")
    return simple_data


def export_to_csv(df: pd.DataFrame, output_path: str = "prompts.csv"):
    """Export data to CSV format."""
    # Create a simplified DataFrame for CSV export
    csv_df = df[["text", "is_injection"]].copy()
    csv_df["is_injection"] = csv_df["is_injection"].astype(
        int
    )  # Convert bool to int for CSV

    csv_df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)

    print(f"Exported {len(csv_df)} prompts to {output_path}")
    return csv_df


def export_to_excel(df: pd.DataFrame, output_path: str = "prompts.xlsx"):
    """Export data to Excel format."""
    # Create a simplified DataFrame for Excel export
    excel_df = df[["text", "is_injection"]].copy()

    excel_df.to_excel(output_path, index=False)

    print(f"Exported {len(excel_df)} prompts to {output_path}")
    return excel_df


def export_statistics(df: pd.DataFrame):
    """Print data statistics."""
    total = len(df)
    injections = df["is_injection"].sum()
    safe = total - injections

    print("\n=== Data Statistics ===")
    print(f"Total prompts: {total}")
    print(f"Injection prompts: {int(injections)} ({injections / total * 100:.1f}%)")
    print(f"Safe prompts: {int(safe)} ({safe / total * 100:.1f}%)")

    # Additional statistics
    print("\nText length statistics:")
    df["text_length"] = df["text"].str.len()
    print(f"  Average length: {df['text_length'].mean():.1f} characters")
    print(f"  Min length: {df['text_length'].min()} characters")
    print(f"  Max length: {df['text_length'].max()} characters")

    return {
        "total": int(total),
        "injections": int(injections),
        "safe": int(safe),
        "injection_percentage": injections / total * 100,
        "safe_percentage": safe / total * 100,
    }


def export_training_data(df: pd.DataFrame, output_path: str = "training_data.txt"):
    """Export in a simple format for training."""
    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            label = "INJECTION" if row["is_injection"] else "SAFE"
            f.write(f"{label}\t{row['text']}\n")

    print(f"Exported {len(df)} prompts to training format: {output_path}")


def export_to_parquet_split(
    df: pd.DataFrame,
    output_dir: str = "data",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
):
    """Export data as train/val/test parquet splits.

    NOTE: Static split files are deprecated. Modern training uses
    merged.parquet with dynamic splits.
    """
    print("NOTE: Creating static split files which are deprecated.")
    print("   Modern training uses merged.parquet with dynamic splits.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    # Save splits
    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"
    test_path = output_dir / "test.parquet"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print(f"\nExported splits to {output_dir}:")
    print(f"  Train: {len(train_df)} samples -> {train_path}")
    print(f"  Validation: {len(val_df)} samples -> {val_path}")
    print(f"  Test: {len(test_df)} samples -> {test_path}")

    return train_path, val_path, test_path


def main():
    parser = argparse.ArgumentParser(
        description="Export prompt injection data from parquet"
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv", "excel", "stats", "training", "parquet-split"],
        default="json",
        help="Export format",
    )
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument(
        "--parquet",
        type=str,
        default="data/merged.parquet",
        help="Input parquet file path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for parquet splits",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.8, help="Training set ratio for splits"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.1, help="Validation set ratio for splits"
    )

    args = parser.parse_args()

    # Set default output filename based on format
    if not args.output:
        if args.format == "json":
            args.output = "prompts.json"
        elif args.format == "csv":
            args.output = "prompts.csv"
        elif args.format == "excel":
            args.output = "prompts.xlsx"
        elif args.format == "training":
            args.output = "training_data.txt"

    # Load data from parquet
    try:
        df = load_from_parquet(args.parquet)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nYou can:")
        print("  1. Run the migration script: python scripts/migrate_to_parquet.py")
        print("  2. Specify a different parquet file: --parquet /path/to/data.parquet")
        print("  3. Check if the package includes data files")
        return 1

    # Export based on format
    if args.format == "json":
        export_to_json(df, args.output)
    elif args.format == "csv":
        export_to_csv(df, args.output)
    elif args.format == "excel":
        export_to_excel(df, args.output)
    elif args.format == "stats":
        export_statistics(df)
    elif args.format == "training":
        export_training_data(df, args.output)
    elif args.format == "parquet-split":
        export_to_parquet_split(df, args.output_dir, args.train_ratio, args.val_ratio)

    # Always show statistics (except for stats format itself)
    if args.format != "stats":
        export_statistics(df)

    return 0


if __name__ == "__main__":
    exit(main())
