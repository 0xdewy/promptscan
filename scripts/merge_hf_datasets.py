#!/usr/bin/env python3
"""
Merge downloaded HuggingFace datasets into a single processed.parquet file.

Canonical schema: id, text, is_injection, source
- Deduplicates by text content (case-insensitive, stripped)
- Maps various label column names to boolean is_injection
"""

import sys
import uuid
from pathlib import Path

import pandas as pd


# Datasets that contain only safe prompts (no injection label column).
# All rows from these sources are labeled is_injection=False.
SAFE_ONLY_SOURCES = {
    "MohamedRashad_ChatGPT-prompts",
    "pvduy_70k_evol_code_prompts",
    "marketeam_marketing_user_prompts_unfiltered",
    "tatsu-lab_alpaca",
}

# Common label column names and their boolean interpretations
LABEL_COLUMNS = [
    "is_injection",
    "label",
    "labels",
    "injection",
    "prompt_injection",
    "malicious",
    "is_malicious",
    "is_safe",
    "safe",
    "attack",
    "is_attack",
]

# Label value mappings to boolean (is_injection=True means it's an injection)
# Keys are lowercase string representations
LABEL_MAPPINGS = {
    # True values (injection)
    True: True,
    1: True,
    "1": True,
    "true": True,
    "True": True,
    "TRUE": True,
    "injection": True,
    "INJECTION": True,
    "malicious": True,
    "MALICIOUS": True,
    "attack": True,
    "ATTACK": True,
    "positive": True,
    "pos": True,
    "yes": True,
    "YES": True,
    "y": True,
    # False values (safe)
    False: False,
    0: False,
    "0": False,
    "false": False,
    "False": False,
    "FALSE": False,
    "safe": False,
    "SAFE": False,
    "benign": False,
    "BENIGN": False,
    "normal": False,
    "NORMAL": False,
    "negative": False,
    "neg": False,
    "no": False,
    "NO": False,
    "n": False,
}


def find_label_column(df: pd.DataFrame) -> str:
    """Find the label column in a dataframe."""
    for col in LABEL_COLUMNS:
        if col in df.columns:
            return col
    # Try case-insensitive match
    cols_lower = {c.lower(): c for c in df.columns}
    for col in LABEL_COLUMNS:
        if col.lower() in cols_lower:
            return cols_lower[col.lower()]
    raise ValueError(f"No label column found. Columns: {list(df.columns)}")


def find_text_column(df: pd.DataFrame) -> str:
    """Find the text column in a dataframe."""
    candidates = ["text", "prompt", "human_prompt", "content", "input", "user_input", "instruction", "query", "message", "sentence"]
    for col in candidates:
        if col in df.columns:
            return col
    cols_lower = {c.lower(): c for c in df.columns}
    for col in candidates:
        if col.lower() in cols_lower:
            return cols_lower[col.lower()]
    raise ValueError(f"No text column found. Columns: {list(df.columns)}")


def map_label(value) -> bool:
    """Convert a label value to boolean (is_injection)."""
    if pd.isna(value):
        raise ValueError(f"Label value is NaN")

    # Handle various types
    if isinstance(value, (bool, int, float)):
        if isinstance(value, float):
            return bool(int(value))
        return bool(value)

    # String
    v = str(value).strip().lower()
    if v in LABEL_MAPPINGS:
        return LABEL_MAPPINGS[v]

    # Check if it's a numeric string
    try:
        return bool(int(v))
    except ValueError:
        pass

    raise ValueError(f"Unknown label value: {value!r}")


def process_dataset(filepath: Path, source_name: str) -> pd.DataFrame:
    """Process a single dataset file into canonical format."""
    df = pd.read_parquet(filepath)
    print(f"  Processing {filepath.name} ({len(df)} rows)...")

    # Find text column (required for all datasets)
    text_col = find_text_column(df)

    # Safe-only datasets: no label column, all prompts are safe
    if source_name in SAFE_ONLY_SOURCES:
        records = []
        for _, row in df.iterrows():
            text = str(row[text_col]).strip()
            if text:
                records.append({
                    "id": str(uuid.uuid4()),
                    "text": text,
                    "is_injection": False,
                    "source": source_name,
                })
        print(f"    Safe-only source: {len(records)} safe prompts")
        return pd.DataFrame(records)

    label_col = find_label_column(df)

    # SPECIAL HANDLING: imoxto datasets label attack SUCCESS, not attack PRESENCE
    # - label=1 means the injection attack SUCCEEDED (keep as injection)
    # - label=0 means the injection attack FAILED (discard - confusing training data)
    # All texts in imoxto are injection attempts, but we only want successful ones
    if source_name.startswith("imoxto_"):
        original_count = len(df)
        # Keep only successful attacks (label=1)
        df = df[df[label_col] == 1]
        print(f"    Imoxto filter: {original_count} -> {len(df)} (keeping successful attacks only)")
        
        records = []
        for _, row in df.iterrows():
            text = str(row[text_col]).strip()
            if len(text) > 0:
                records.append(
                    {
                        "id": str(uuid.uuid4()),
                        "text": text,
                        "is_injection": True,  # All successful attacks are injections
                        "source": source_name,
                    }
                )
        return pd.DataFrame(records)

    # Standard processing for other datasets
    records = []
    errors = 0

    for _, row in df.iterrows():
        try:
            text = str(row[text_col]).strip()
            if len(text) == 0:
                continue

            label_value = row[label_col]
            is_injection = map_label(label_value)

            records.append(
                {
                    "id": str(uuid.uuid4()),
                    "text": text,
                    "is_injection": is_injection,
                    "source": source_name,
                }
            )
        except Exception:
            errors += 1

    if errors > 0:
        print(f"    Warning: {errors} rows skipped due to errors")

    return pd.DataFrame(records)


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate by text content (case-insensitive, stripped)."""
    before = len(df)
    df["text_normalized"] = df["text"].str.lower().str.strip()
    df = df.drop_duplicates(subset=["text_normalized"], keep="first")
    df = df.drop(columns=["text_normalized"])
    after = len(df)
    print(f"  Deduplication: {before} -> {after} (removed {before - after} duplicates)")
    return df


def main():
    script_dir = Path(__file__).parent
    hf_dir = script_dir.parent / "data" / "hf_datasets"
    output_path = script_dir.parent / "data" / "processed.parquet"

    if not hf_dir.exists():
        print(f"Error: hf_datasets directory not found at {hf_dir}")
        print("  Run scripts/download_hf_datasets.py first")
        sys.exit(1)

    parquet_files = sorted(hf_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"Error: No parquet files found in {hf_dir}")
        sys.exit(1)

    print(f"Found {len(parquet_files)} datasets to merge")
    print()

    all_records = []
    skipped_files = []

    for filepath in parquet_files:
        source_name = filepath.stem  # filename without extension
        try:
            df = process_dataset(filepath, source_name)
            all_records.append(df)
        except Exception as e:
            print(f"  ERROR: {e}")
            skipped_files.append((filepath.name, str(e)))

    if not all_records:
        print("Error: No datasets could be processed")
        sys.exit(1)

    print()
    print("Concatenating datasets...")
    combined = pd.concat(all_records, ignore_index=True)
    print(f"  Total rows before dedup: {len(combined):,}")

    print()
    print("Deduplicating...")
    combined = deduplicate(combined)

    print()
    print("=" * 60)
    print("MERGE SUMMARY")
    print("=" * 60)
    print(f"Total rows: {len(combined):,}")
    print(
        f"  Injections: {combined['is_injection'].sum():,} ({combined['is_injection'].mean() * 100:.1f}%)"
    )
    print(
        f"  Safe: {(~combined['is_injection']).sum():,} ({(1 - combined['is_injection'].mean()) * 100:.1f}%)"
    )
    print()
    print("Rows per source:")
    for source, count in combined["source"].value_counts().items():
        print(f"  {source}: {count:,}")

    if skipped_files:
        print()
        print(f"Skipped files: {len(skipped_files)}")
        for name, err in skipped_files:
            print(f"  {name}: {err}")

    print()
    print(f"Saving to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)
    print(f"Saved {len(combined):,} rows")

    # Verify against canonical
    print()
    print("Verifying output schema...")
    canonical_cols = ["id", "text", "is_injection", "source"]
    actual_cols = list(combined.columns)
    if all(c in actual_cols for c in canonical_cols):
        print(f"  Schema OK: {actual_cols}")
    else:
        print(
            f"  WARNING: Missing columns. Expected {canonical_cols}, got {actual_cols}"
        )


if __name__ == "__main__":
    main()
