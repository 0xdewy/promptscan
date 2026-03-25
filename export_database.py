#!/usr/bin/env python3
"""
Export prompt injection database to various formats.
"""

import sqlite3
import json
import csv
import pandas as pd
from typing import List, Dict, Any
import argparse


def export_to_json(db_path: str = "prompts.db", output_path: str = "prompts.json"):
    """Export database to JSON format."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT text, is_injection FROM prompts")
    rows = cursor.fetchall()

    data = []
    for text, is_injection in rows:
        item = {"text": text, "is_injection": bool(is_injection)}

        data.append(item)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    conn.close()
    print(f"Exported {len(data)} prompts to {output_path}")
    return data


def export_to_csv(db_path: str = "prompts.db", output_path: str = "prompts.csv"):
    """Export database to CSV format."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT text, is_injection FROM prompts")
    rows = cursor.fetchall()

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "is_injection"])

        for row in rows:
            writer.writerow(row)

    conn.close()
    print(f"Exported {len(rows)} prompts to {output_path}")
    return rows


def export_to_sql(db_path: str = "prompts.db", output_path: str = "prompts_backup.db"):
    """Create a backup SQL file of the database."""
    conn = sqlite3.connect(db_path)

    with open(output_path, "w") as f:
        for line in conn.iterdump():
            f.write(f"{line}\n")

    conn.close()
    print(f"Exported database to SQL backup: {output_path}")


def export_to_excel(db_path: str = "prompts.db", output_path: str = "prompts.xlsx"):
    """Export database to Excel format."""
    conn = sqlite3.connect(db_path)

    query = "SELECT text, is_injection FROM prompts"

    df = pd.read_sql_query(query, conn)
    df["is_injection"] = df["is_injection"].astype(bool)

    df.to_excel(output_path, index=False)

    conn.close()
    print(f"Exported {len(df)} prompts to {output_path}")
    return df


def export_statistics(db_path: str = "prompts.db"):
    """Print database statistics."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Total prompts
    cursor.execute("SELECT COUNT(*) FROM prompts")
    total = cursor.fetchone()[0]

    # Injection vs safe
    cursor.execute("SELECT COUNT(*) FROM prompts WHERE is_injection = 1")
    injections = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM prompts WHERE is_injection = 0")
    safe = cursor.fetchone()[0]

    print("\n=== Database Statistics ===")
    print(f"Total prompts: {total}")
    print(f"Injection prompts: {injections} ({injections / total * 100:.1f}%)")
    print(f"Safe prompts: {safe} ({safe / total * 100:.1f}%)")

    conn.close()


def export_training_data(
    db_path: str = "prompts.db", output_path: str = "training_data.txt"
):
    """Export in a simple format for training."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT text, is_injection FROM prompts")
    rows = cursor.fetchall()

    with open(output_path, "w", encoding="utf-8") as f:
        for text, is_injection in rows:
            label = "INJECTION" if is_injection else "SAFE"
            f.write(f"{label}\t{text}\n")

    conn.close()
    print(f"Exported {len(rows)} prompts to training format: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export prompt injection database")
    parser.add_argument(
        "--format",
        choices=["json", "csv", "excel", "sql", "stats", "training"],
        default="json",
        help="Export format",
    )
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--db", type=str, default="prompts.db", help="Database path")

    args = parser.parse_args()

    # Set default output filename based on format
    if not args.output:
        if args.format == "json":
            args.output = "prompts.json"
        elif args.format == "csv":
            args.output = "prompts.csv"
        elif args.format == "excel":
            args.output = "prompts.xlsx"
        elif args.format == "sql":
            args.output = "prompts_backup.db"
        elif args.format == "training":
            args.output = "training_data.txt"

    if args.format == "json":
        export_to_json(args.db, args.output)
    elif args.format == "csv":
        export_to_csv(args.db, args.output)
    elif args.format == "excel":
        export_to_excel(args.db, args.output)
    elif args.format == "sql":
        export_to_sql(args.db, args.output)
    elif args.format == "stats":
        export_statistics(args.db)
    elif args.format == "training":
        export_training_data(args.db, args.output)

    # Always show statistics
    if args.format != "stats":
        export_statistics(args.db)


if __name__ == "__main__":
    main()
