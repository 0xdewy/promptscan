#!/usr/bin/env python3
"""Read and display parquet files in the data directory."""

import argparse
import sys
from pathlib import Path

import pandas as pd


def read_parquet(
    path: str, limit: int | None = None, show_sources: bool = False
) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if limit:
        df = df.head(limit)
    return df


def print_stats(df: pd.DataFrame) -> None:
    total = len(df)
    print(f"Total rows: {total}")
    print(f"Columns: {df.columns.tolist()}")

    if "is_injection" in df.columns:
        inj = df["is_injection"].value_counts()
        print(f"is_injection distribution:")
        for val, count in inj.items():
            print(f"  {val}: {count} ({count / total * 100:.1f}%)")

    if "source" in df.columns:
        sources = df["source"].value_counts()
        print(f"Sources:")
        for src, count in sources.items():
            print(f"  {src}: {count}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Read and display parquet files")
    parser.add_argument(
        "file", nargs="?", help="Path to parquet file (default: data/merged.parquet)"
    )
    parser.add_argument("--limit", "-n", type=int, help="Limit number of rows shown")
    parser.add_argument("--stats", "-s", action="store_true", help="Show statistics")
    args = parser.parse_args()

    path = args.file or "data/merged.parquet"
    if not Path(path).exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        return 1

    try:
        df = read_parquet(path, args.limit)
    except Exception as e:
        print(f"Error reading parquet: {e}", file=sys.stderr)
        return 1

    if args.stats:
        print_stats(df)
        print()

    pd.set_option("display.max_colwidth", 80)
    pd.set_option("display.width", 120)
    print(df.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
