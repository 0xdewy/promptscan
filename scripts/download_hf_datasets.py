#!/usr/bin/env python3
"""
Download HuggingFace datasets from datasets.txt and save as individual parquet files.
"""

import sys
import os
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' package required. Install with: uv add datasets")
    sys.exit(1)


def download_dataset(dataset_url: str, output_dir: Path) -> Path:
    """
    Download a HuggingFace dataset and save as parquet.

    Args:
        dataset_url: HuggingFace dataset URL (e.g., https://huggingface.co/datasets/cgoosen/...)
        output_dir: Directory to save parquet files

    Returns:
        Path to saved parquet file
    """
    from huggingface_hub import HfApi

    # Extract repo_id from URL
    # URL format: https://huggingface.co/datasets/cgoosen/prompt_injection_ctf_dataset_2
    parts = dataset_url.strip().rstrip("/").split("/")
    repo_id = "/".join(parts[-2:])  # e.g., cgoosen/prompt_injection_ctf_dataset_2

    # Create safe filename
    safe_name = repo_id.replace("/", "_")
    output_path = output_dir / f"{safe_name}.parquet"

    if output_path.exists():
        print(f"  Already exists, skipping: {output_path.name}")
        return output_path

    print(f"  Loading dataset from HuggingFace...")
    try:
        dataset = load_dataset(repo_id)
    except Exception as e:
        print(f"  Error loading dataset: {e}")
        raise

    # Handle DatasetDict (multiple splits) vs Dataset (single)
    if hasattr(dataset, "keys"):
        # It's a DatasetDict - choose the biggest split or 'train'
        splits = list(dataset.keys())
        print(f"  Available splits: {splits}")

        if "train" in splits:
            split_name = "train"
        else:
            # Pick the largest split
            split_name = max(splits, key=lambda s: len(dataset[s]))
        print(f"  Using split: {split_name}")
        df = dataset[split_name].to_pandas()
    else:
        # It's a single Dataset
        df = dataset.to_pandas()

    print(f"  Dataset has {len(df)} rows and columns: {list(df.columns)}")

    # Save to parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"  Saved to: {output_path}")

    return output_path


def main():
    script_dir = Path(__file__).parent
    datasets_file = script_dir.parent / "datasets.txt"
    output_dir = script_dir.parent / "data" / "hf_datasets"

    if not datasets_file.exists():
        print(f"Error: datasets.txt not found at {datasets_file}")
        sys.exit(1)

    # Read dataset URLs
    with open(datasets_file, "r") as f:
        urls = [line.strip() for line in f if line.strip()]

    if not urls:
        print("Error: datasets.txt is empty")
        sys.exit(1)

    print(f"Found {len(urls)} datasets to download")
    print(f"Output directory: {output_dir}")
    print()

    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    errors = []

    for i, url in enumerate(urls, 1):
        print(f"[{i}/{len(urls)}] {url}")
        try:
            path = download_dataset(url, output_dir)
            results.append((url, path, None))
        except Exception as e:
            print(f"  FAILED: {e}")
            errors.append((url, str(e)))

    print()
    print("=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"Successful: {len(results)}")
    print(f"Failed: {len(errors)}")

    if results:
        print("\nDownloaded files:")
        for url, path, _ in results:
            size_mb = path.stat().st_size / 1024 / 1024
            print(f"  {path.name} ({size_mb:.1f} MB)")

    if errors:
        print("\nFailed downloads:")
        for url, err in errors:
            print(f"  {url}: {err}")

    print()
    print(f"All files saved to: {output_dir}")


if __name__ == "__main__":
    main()
