#!/usr/bin/env python3
"""
Download conversational safe datasets from HuggingFace and add to training data.

Adds natural human-conversational text labeled as SAFE, which fixes the root
cause of false positives: the training set had no examples of normal instructional
language labeled SAFE — only code prompts and formatted Alpaca-style examples.

Datasets added:
  - Anthropic/hh-rlhf       (human preference conversations)
  - OpenAssistant/oasst1     (human assistant dialogues)
  - databricks-dolly-15k     (diverse natural instructions)
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from promptscan.parquet_store import ParquetDataStore

MAX_PER_DATASET = 20_000


def extract_hh_rlhf(max_samples: int = MAX_PER_DATASET) -> list[str]:
    """Extract human turns from Anthropic/hh-rlhf."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  Error: 'datasets' package required. Install with: uv add datasets")
        return []

    print("  Loading Anthropic/hh-rlhf...")
    try:
        ds = load_dataset("Anthropic/hh-rlhf", split="train")
    except Exception as e:
        print(f"  Failed to load: {e}")
        return []

    texts = []
    for item in ds:
        chosen = item.get("chosen", "")
        # Format: "\n\nHuman: ...\n\nAssistant: ...\n\nHuman: ..."
        # Split on turn markers and extract human messages
        segments = re.split(r"\n\nHuman: |\n\nAssistant: ", chosen)
        # Odd-indexed segments (1, 3, 5...) are human turns
        for i in range(1, len(segments), 2):
            text = segments[i].strip()
            if 3 <= len(text) <= 2000:
                texts.append(text)
                if len(texts) >= max_samples:
                    break
        if len(texts) >= max_samples:
            break

    print(f"  Extracted {len(texts):,} human turns")
    return texts[:max_samples]


def extract_oasst1(max_samples: int = MAX_PER_DATASET) -> list[str]:
    """Extract English user messages from OpenAssistant/oasst1."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  Error: 'datasets' package required.")
        return []

    print("  Loading OpenAssistant/oasst1...")
    try:
        ds = load_dataset("OpenAssistant/oasst1", split="train")
    except Exception as e:
        print(f"  Failed to load: {e}")
        return []

    texts = []
    for item in ds:
        if item.get("role") == "prompter" and item.get("lang") == "en":
            text = item.get("text", "").strip()
            if 3 <= len(text) <= 2000:
                texts.append(text)
        if len(texts) >= max_samples:
            break

    print(f"  Extracted {len(texts):,} user messages")
    return texts[:max_samples]


def extract_dolly(max_samples: int = MAX_PER_DATASET) -> list[str]:
    """Extract instructions from databricks/databricks-dolly-15k."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  Error: 'datasets' package required.")
        return []

    print("  Loading databricks/databricks-dolly-15k...")
    try:
        ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    except Exception as e:
        print(f"  Failed to load: {e}")
        return []

    texts = []
    for item in ds:
        text = item.get("instruction", "").strip()
        if 3 <= len(text) <= 2000:
            texts.append(text)
        if len(texts) >= max_samples:
            break

    print(f"  Extracted {len(texts):,} instructions")
    return texts[:max_samples]


DATASETS = [
    ("Anthropic/hh-rlhf", "anthropic_hh_rlhf", extract_hh_rlhf),
    ("OpenAssistant/oasst1", "openassistant_oasst1", extract_oasst1),
    ("databricks/databricks-dolly-15k", "databricks_dolly_15k", extract_dolly),
]


def main():
    parquet_path = "data/merged.parquet"
    store = ParquetDataStore(parquet_path)

    total_added = 0
    total_skipped = 0

    for dataset_name, source_name, extractor in DATASETS:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        texts = extractor()
        if not texts:
            print(f"  No texts extracted, skipping.")
            continue

        prompts = [
            {"text": text, "is_injection": False, "source": source_name}
            for text in texts
        ]

        added_ids, skipped = store.add_prompts_batch(prompts)
        total_added += len(added_ids)
        total_skipped += skipped
        print(f"  Added: {len(added_ids):,}, Skipped: {skipped:,}")

    print(f"\n{'='*60}")
    print(f"Total added: {total_added:,}, Total skipped: {total_skipped:,}")
    stats = store.get_statistics()
    print(
        f"Database: {stats['total']:,} total "
        f"({stats['injections']:,} inj / {stats['safe']:,} safe)"
    )


if __name__ == "__main__":
    main()
