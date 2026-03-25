#!/usr/bin/env python3
"""
Extract prompts and queries from output.json and insert them as safe prompts.
"""

import json
import sqlite3
import sys
from typing import List, Dict, Any


def extract_texts_from_json(json_path: str = "output.json") -> List[str]:
    """Extract all text from prompts and queries in output.json."""

    print(f"Reading {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_texts = []

    # Process each item in the array
    for item in data:
        # Extract prompt text
        if "prompt" in item and "text" in item["prompt"]:
            prompt_text = item["prompt"]["text"].strip()
            if prompt_text:
                all_texts.append(prompt_text)

        # Extract query texts
        if "queries" in item:
            for query in item["queries"]:
                if "text" in query:
                    query_text = query["text"].strip()
                    if query_text:
                        all_texts.append(query_text)

    print(f"Extracted {len(all_texts)} texts from {json_path}")

    # Show some samples
    print("\nSample texts:")
    for i, text in enumerate(all_texts[:5], 1):
        preview = text[:100] + "..." if len(text) > 100 else text
        print(f"  {i}. {preview}")

    return all_texts


def insert_texts_as_safe_prompts(texts: List[str], db_path: str = "prompts.db"):
    """Insert extracted texts as safe prompts into database."""

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Ensure table exists
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS prompts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT NOT NULL,
        is_injection BOOLEAN NOT NULL
    )
    """)

    # Insert texts
    inserted_count = 0
    skipped_count = 0

    for text in texts:
        # Check if text already exists (simple exact match)
        cursor.execute("SELECT COUNT(*) FROM prompts WHERE text = ?", (text,))
        exists = cursor.fetchone()[0] > 0

        if not exists:
            cursor.execute(
                "INSERT INTO prompts (text, is_injection) VALUES (?, ?)",
                (text, False),  # All are safe prompts
            )
            inserted_count += 1

            # Show progress
            if inserted_count % 50 == 0:
                print(f"  Inserted {inserted_count} texts...")
        else:
            skipped_count += 1

    conn.commit()

    # Get updated statistics
    cursor.execute("SELECT COUNT(*) FROM prompts WHERE is_injection = 1")
    total_injections = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM prompts WHERE is_injection = 0")
    total_safe = cursor.fetchone()[0]

    conn.close()

    print(f"\nInsertion complete:")
    print(f"  - Inserted: {inserted_count} new safe prompts")
    print(f"  - Skipped: {skipped_count} duplicates")
    print(f"  - Total prompts in database: {total_injections + total_safe}")
    print(f"    • Injections: {total_injections}")
    print(f"    • Safe prompts: {total_safe}")

    return inserted_count


def analyze_text_lengths(texts: List[str]):
    """Analyze text lengths for database optimization."""

    lengths = [len(text) for text in texts]

    print("\nText length analysis:")
    print(f"  - Total texts: {len(texts)}")
    print(f"  - Average length: {sum(lengths) / len(lengths):.0f} characters")
    print(f"  - Min length: {min(lengths)} characters")
    print(f"  - Max length: {max(lengths)} characters")

    # Count by length categories
    short = len([l for l in lengths if l < 100])
    medium = len([l for l in lengths if 100 <= l < 500])
    long = len([l for l in lengths if l >= 500])

    print(f"  - Short (<100 chars): {short} texts")
    print(f"  - Medium (100-500 chars): {medium} texts")
    print(f"  - Long (≥500 chars): {long} texts")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Import texts from output.json as safe prompts"
    )
    parser.add_argument(
        "--json", type=str, default="output.json", help="Path to output.json"
    )
    parser.add_argument("--db", type=str, default="prompts.db", help="Database path")
    parser.add_argument(
        "--analyze", action="store_true", help="Only analyze, don't insert"
    )
    parser.add_argument("--limit", type=int, help="Limit number of texts to insert")

    args = parser.parse_args()

    # Extract texts
    texts = extract_texts_from_json(args.json)

    # Analyze
    analyze_text_lengths(texts)

    # Limit if specified
    if args.limit and args.limit < len(texts):
        print(f"\nLimiting to {args.limit} texts (from {len(texts)})")
        texts = texts[: args.limit]

    if not args.analyze:
        # Insert into database
        print(f"\nInserting texts into database {args.db}...")
        inserted = insert_texts_as_safe_prompts(texts, args.db)

        if inserted > 0:
            print(f"\n✅ Successfully added {inserted} safe prompts to database!")
            print("\nYou can now retrain the model:")
            print("  python detector.py train")
        else:
            print("\n⚠️ No new texts were inserted (all were duplicates)")
    else:
        print(
            "\nAnalysis complete. Use without --analyze flag to insert into database."
        )


if __name__ == "__main__":
    main()
