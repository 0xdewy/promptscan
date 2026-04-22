#!/usr/bin/env python3
"""
Data validation script for promptscan merged.parquet dataset.

Validates data quality and generates a report of issues found.
Does NOT modify the data - only analyzes and reports.

Usage:
    python scripts/validate_data.py
"""

import random
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd


def load_data(path: str = "data/merged.parquet") -> pd.DataFrame:
    """Load the merged parquet dataset."""
    print(f"Loading data from {path}...")
    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} rows with {len(df.columns)} columns")
    return df


def check_nulls(df: pd.DataFrame) -> Dict[str, Any]:
    """Check for null/missing values in all columns."""
    print("\n=== NULL VALUE ANALYSIS ===")
    null_counts = df.isnull().sum()
    null_pcts = (null_counts / len(df) * 100).round(2)

    results = {
        "total_rows": len(df),
        "columns_with_nulls": [],
    }

    for col in df.columns:
        count = null_counts[col]
        pct = null_pcts[col]
        if count > 0:
            results["columns_with_nulls"].append(
                {"column": col, "null_count": int(count), "null_pct": float(pct)}
            )
            print(f"  {col}: {count:,} ({pct}%)")

    if not results["columns_with_nulls"]:
        print("  No null values found!")

    return results


def check_text_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Check text quality issues."""
    print("\n=== TEXT QUALITY ANALYSIS ===")
    results = {
        "empty_texts": 0,
        "very_short_texts": [],
        "duplicate_texts": [],
        "encoding_issues": [],
    }

    empty_mask = df["text"].str.strip() == ""
    results["empty_texts"] = int(empty_mask.sum())
    print(f"  Empty/whitespace-only texts: {results['empty_texts']}")

    very_short_mask = df["text"].str.len() < 5
    very_short = df[very_short_mask][["id", "text", "is_injection"]].copy()
    results["very_short_texts"] = [
        {"id": r["id"], "text": r["text"], "is_injection": r["is_injection"]}
        for _, r in very_short.iterrows()
    ]
    print(f"  Very short texts (< 5 chars): {len(very_short)}")
    if len(very_short) > 0:
        for _, row in very_short.head(5).iterrows():
            print(f"    - [{row['is_injection']}] '{row['text']}'")

    duplicate_texts = df[df.duplicated(subset=["text"], keep=False)]
    results["duplicate_texts"] = len(duplicate_texts)
    print(f"  Duplicate texts (by content): {len(duplicate_texts)}")

    return results


def check_labels(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze label distribution and potential mislabelings."""
    print("\n=== LABEL DISTRIBUTION ===")
    results = {
        "total": len(df),
        "injection_true": int(df["is_injection"].sum()),
        "injection_false": int((~df["is_injection"]).sum()),
        "injection_rate": float(df["is_injection"].mean()),
        "by_source": {},
        "potential_mislabeled": [],
    }

    print(
        f"  is_injection=True:  {results['injection_true']:,} ({results['injection_rate'] * 100:.2f}%)"
    )
    print(
        f"  is_injection=False: {results['injection_false']:,} ({(1 - results['injection_rate']) * 100:.2f}%)"
    )

    print("\n=== INJECTION RATE BY SOURCE (top 15) ===")
    source_stats = (
        df.groupby("source")
        .agg(injections=("is_injection", "sum"), total=("is_injection", "count"))
        .reset_index()
    )
    source_stats["rate"] = (source_stats["injections"] / source_stats["total"]).round(3)
    source_stats = source_stats.sort_values("total", ascending=False)
    results["by_source"] = source_stats.to_dict("records")

    for _, row in source_stats.head(15).iterrows():
        print(
            f"  {row['source'][:60]}: {int(row['injections'])}/{int(row['total'])} ({row['rate']:.1%})"
        )

    SUSPICIOUS_KEYWORDS = [
        "secret",
        "don't tell",
        "never tell",
        "password",
        "api_key",
        "apikey",
        "hidden",
        "ignore previous",
        "ignore instructions",
        "disregard",
        "system prompt",
        "admin mode",
        "developer mode",
        "jailbreak",
    ]

    print(
        "\n=== POTENTIAL MISLABELED TEXTS (False but contain suspicious keywords) ==="
    )
    pattern = "|".join(SUSPICIOUS_KEYWORDS)
    mask = (~df["is_injection"]) & df["text"].str.lower().str.contains(
        pattern, na=False
    )
    suspicious = df[mask].sample(n=min(10, mask.sum()), random_state=42)
    results["potential_mislabeled"] = [
        {"id": r["id"], "text": r["text"][:200], "source": r["source"]}
        for _, r in suspicious.iterrows()
    ]
    for _, row in suspicious.iterrows():
        print(f"  [{row['source'][:40]}] {row['text'][:100]}...")

    return results


def check_sources(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze and flag suspicious sources."""
    print("\n=== SOURCE ANALYSIS ===")
    results = {
        "total_unique_sources": int(df["source"].nunique()),
        "suspicious_sources": [],
        "major_sources": [],
    }

    print(f"  Total unique sources: {results['total_unique_sources']}")

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

    suspicious_mask = df["source"].str.contains("|".join(SUSPICIOUS_PATTERNS), na=False)
    suspicious_sources = df[suspicious_mask]["source"].unique()
    results["suspicious_sources"] = list(suspicious_sources)
    print(f"  Suspicious sources (internal/odd): {len(suspicious_sources)}")
    for src in suspicious_sources[:10]:
        count = len(df[df["source"] == src])
        print(f"    - {src}: {count} rows")

    source_counts = df["source"].value_counts()
    major_sources = source_counts[source_counts >= 1000].index.tolist()
    results["major_sources"] = major_sources
    print(f"\n  Major sources (>= 1000 rows): {len(major_sources)}")
    for src in major_sources[:5]:
        print(f"    - {src}: {source_counts[src]:,} rows")

    return results


def generate_validation_samples(
    df: pd.DataFrame, n_per_class: int = 20
) -> pd.DataFrame:
    """Generate stratified random samples for manual label review."""
    print(f"\n=== GENERATING VALIDATION SAMPLES ({n_per_class} per class) ===")

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
                    "text_length": len(row["text"]),
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


def generate_report(
    null_results: Dict,
    text_results: Dict,
    label_results: Dict,
    source_results: Dict,
    output_path: str = "data/validation_report.md",
) -> None:
    """Generate a markdown validation report."""
    print(f"\n=== GENERATING REPORT ===")

    report = f"""# Data Validation Report

Generated from `data/merged.parquet`

## Summary Statistics

- **Total Rows**: {null_results["total_rows"]:,}
- **Total Columns**: 11
- **Unique Sources**: {source_results["total_unique_sources"]:,}

## Null Value Analysis

| Column | Null Count | Null % |
|--------|------------|--------|
"""

    for col_info in null_results["columns_with_nulls"]:
        report += f"| {col_info['column']} | {col_info['null_count']:,} | {col_info['null_pct']}% |\n"

    if not null_results["columns_with_nulls"]:
        report += "| _None_ | 0 | 0% |\n"

    report += f"""
## Text Quality Issues

- **Empty texts**: {text_results["empty_texts"]}
- **Very short texts (< 5 chars)**: {len(text_results["very_short_texts"])}
- **Duplicate texts**: {text_results["duplicate_texts"]}

## Label Distribution

- **is_injection=True**: {label_results["injection_true"]:,} ({label_results["injection_rate"] * 100:.2f}%)
- **is_injection=False**: {label_results["injection_false"]:,} ({(1 - label_results["injection_rate"]) * 100:.2f}%)

### Injection Rate by Source (Top 15)

| Source | Injections | Total | Rate |
|--------|------------|-------|------|
"""

    for src in label_results["by_source"][:15]:
        report += f"| {src['source'][:50]} | {int(src['injections'])} | {int(src['total'])} | {src['rate']:.1%} |\n"

    report += f"""
## Source Analysis

- **Total unique sources**: {source_results["total_unique_sources"]}
- **Major sources (>= 1000 rows)**: {len(source_results["major_sources"])}
- **Suspicious sources (flagged for removal)**: {len(source_results["suspicious_sources"])}

### Suspicious Sources Detected:
"""

    for src in source_results["suspicious_sources"][:20]:
        report += f"- `{src}`\n"

    report += f"""
## Potential Mislabeling Issues

{len(label_results["potential_mislabeled"])} texts flagged as potentially mislabeled.
These are texts labeled `is_injection=False` but contain suspicious keywords.

## Recommendations

1. **Fix text_length**: Currently 97% null - should be computed for all rows
2. **Remove suspicious sources**: Filter out internal paths ({len(source_results["suspicious_sources"])} sources)
3. **Manual label review**: Use `data/validation_samples.csv` to review {40} samples
4. **Drop unused columns**: `original_*` columns are 100% null
5. **Deduplicate**: Remove {text_results["duplicate_texts"]} duplicate text entries

## Next Steps

1. Run `python scripts/clean_data.py` to apply fixes
2. Review `data/validation_samples.csv` and update labels
3. Re-run validation after cleaning to verify fixes
"""

    with open(output_path, "w") as f:
        f.write(report)

    print(f"  Report saved to {output_path}")


def main():
    """Run full validation pipeline."""
    print("=" * 60)
    print("PROMPTSCAN DATA VALIDATION")
    print("=" * 60)

    df = load_data()

    null_results = check_nulls(df)
    text_results = check_text_quality(df)
    label_results = check_labels(df)
    source_results = check_sources(df)

    generate_validation_samples(df, n_per_class=20)

    generate_report(null_results, text_results, label_results, source_results)

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
    print("\nOutputs:")
    print("  - data/validation_samples.csv  (samples for manual review)")
    print("  - data/validation_report.md    (full report)")
    print("\nNext: Review validation_samples.csv, then run:")
    print("  python scripts/clean_data.py")


if __name__ == "__main__":
    main()
