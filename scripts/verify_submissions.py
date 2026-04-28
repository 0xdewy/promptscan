#!/usr/bin/env python3
"""
Verify unverified user submissions using MiniMax API.
Sends each submission to MiniMax for classification; if MiniMax agrees with
the user_label, the entry is inserted into merged.parquet for training.
Clears the unverified file after processing.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from promptscan.feedback_store import ParquetFeedbackStore
from promptscan.parquet_store import ParquetDataStore

load_dotenv()

MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY")
MINIMAX_GROUP_ID = os.environ.get("GROUP_ID")
if not MINIMAX_API_KEY:
    print("Error: MINIMAX_API_KEY not found in .env", file=sys.stderr)
    sys.exit(1)
if not MINIMAX_GROUP_ID:
    print("Error: GROUP_ID not found in .env", file=sys.stderr)
    sys.exit(1)

MINIMAX_API_URL = (
    f"https://api.minimaxi.chat/v1/chat/completions?GroupId={MINIMAX_GROUP_ID}"
)
MODEL = "MiniMax-M2.7"

UNVERIFIED_PATH = project_root / "data" / "unverified_user_submissions.parquet"
MERGED_PATH = project_root / "data" / "merged.parquet"
BATCH_SIZE = 20
MAX_RETRIES = 3


def build_verification_prompt(entries: List[Dict[str, Any]]) -> str:
    texts_block = ""
    for i, entry in enumerate(entries):
        truncated = entry["text"][:800]
        texts_block += f"[{i}] {truncated}\n"

    return f"""You are a prompt injection classifier. For each text below, determine whether it is a prompt injection attack (label=1) or safe/legitimate text (label=0).

A prompt injection is any text that attempts to manipulate, override, or bypass an AI system's instructions. Examples include: ignoring prior instructions, role-playing as an unrestricted AI, extracting system prompts, jailbreaks, or tricking the AI into producing harmful content.

Safe text includes normal questions, instructions, creative writing requests, and legitimate uses.

Texts to classify:
{texts_block}
Return ONLY a JSON array where each element has "index" (the number in brackets) and "label" (0 or 1). Example:
[{{"index": 0, "label": 1}}, {{"index": 1, "label": 0}}]

Return only the JSON array, no other text."""


def extract_json_array(content: str) -> Optional[List[Dict[str, Any]]]:
    content = content.strip()
    content = re.sub(r"<think>.*?", "", content, flags=re.DOTALL).strip()
    content = re.sub(r"^```(?:json)?\n?", "", content)
    content = re.sub(r"\n?```$", "", content).strip()

    try:
        result = json.loads(content)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    json_match = re.search(r"\[.*\]", content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def call_minimax(
    entries: List[Dict[str, Any]],
) -> Dict[int, int]:
    headers = {
        "Authorization": f"Bearer {MINIMAX_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": build_verification_prompt(entries)}
        ],
        "temperature": 0.1,
        "max_tokens": 4000,
    }

    data = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                MINIMAX_API_URL, headers=headers, json=payload, timeout=120
            )
            resp.raise_for_status()
            data = resp.json()
            break
        except requests.exceptions.HTTPError as e:
            if attempt == MAX_RETRIES - 1:
                print(
                    f"  [verify] API failed after {MAX_RETRIES} attempts: {e}",
                    file=sys.stderr,
                )
                raise
            print(f"  [verify] Attempt {attempt + 1} failed: {e} — retrying in 5s...")
            time.sleep(5)

    content = data["choices"][0]["message"]["content"]
    content = re.sub(r"<think>.*?", "", content, flags=re.DOTALL).strip()

    if not content:
        raise ValueError("MiniMax returned empty response")

    results = None
    for attempt in range(3):
        results = extract_json_array(content)
        if results is not None:
            break
        print(f"  [verify] JSON parse attempt {attempt + 1} failed, retrying...")

    if results is None:
        raise ValueError(f"Could not parse MiniMax response: {content[:300]}")

    label_map: Dict[int, int] = {}
    for item in results:
        idx = item.get("index")
        label = item.get("label")
        if idx is not None and label is not None:
            label_map[int(idx)] = int(label)

    return label_map


def main():
    parser = argparse.ArgumentParser(
        description="Verify unverified user submissions via MiniMax API"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without making changes",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Submissions per API call (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--unverified",
        default=str(UNVERIFIED_PATH),
        help="Path to unverified submissions parquet",
    )
    parser.add_argument(
        "--merged",
        default=str(MERGED_PATH),
        help="Path to merged training data parquet",
    )
    args = parser.parse_args()

    unverified_path = Path(args.unverified)
    if not unverified_path.exists():
        print(f"Error: {unverified_path} not found", file=sys.stderr)
        sys.exit(1)

    feedback_store = ParquetFeedbackStore(str(unverified_path))
    prompts_store = ParquetDataStore(str(args.merged))

    entries = feedback_store.get_all_feedback()
    if not entries:
        print("No unverified submissions to process.")
        return

    total = len(entries)
    print(f"Loaded {total} unverified submissions")
    db_stats = prompts_store.get_statistics()
    print(f"Training data currently has {db_stats['total']} prompts")

    stats = {
        "total": total,
        "agreed_inserted": 0,
        "disagreed_skipped": 0,
        "duplicates_skipped": 0,
        "api_errors": 0,
    }

    batches = [
        entries[i : i + args.batch_size] for i in range(0, total, args.batch_size)
    ]

    for batch_num, batch in enumerate(batches, 1):
        print(f"\n--- Batch {batch_num}/{len(batches)} ({len(batch)} entries) ---")

        try:
            minimax_labels = call_minimax(batch)
        except Exception as e:
            print(f"  API error, skipping batch: {e}")
            stats["api_errors"] += len(batch)
            continue

        to_insert: List[Dict[str, Any]] = []

        for i, entry in enumerate(batch):
            text = entry["text"]
            user_label = entry.get("user_label", "")
            user_is_injection = user_label == "INJECTION"

            minimax_label = minimax_labels.get(i)
            if minimax_label is None:
                print(f"  [{i}] No MiniMax label returned, skipping")
                stats["api_errors"] += 1
                continue

            minimax_is_injection = bool(minimax_label)

            if minimax_is_injection == user_is_injection:
                to_insert.append(
                    {
                        "text": text,
                        "is_injection": user_is_injection,
                        "source": "user_feedback_verified",
                    }
                )
            else:
                label_str = "INJECTION" if minimax_is_injection else "SAFE"
                print(
                    f"  [{i}] Disagreed: user={user_label},"
                    f" minimax={label_str} — skipped"
                )
                stats["disagreed_skipped"] += 1

        if to_insert:
            if args.dry_run:
                print(
                    f"  [DRY RUN] Would insert {len(to_insert)} entries"
                )
                stats["agreed_inserted"] += len(to_insert)
            else:
                added_ids, dup_count = prompts_store.add_prompts_batch(to_insert)
                stats["agreed_inserted"] += len(added_ids)
                stats["duplicates_skipped"] += dup_count
                print(
                    f"  Inserted: {len(added_ids)}, Duplicates skipped: {dup_count}"
                )

        if batch_num % 10 == 0:
            print(
                f"  Progress: {min(batch_num * args.batch_size, total)}/{total}"
            )

    print("\n" + "=" * 50)
    print("VERIFICATION COMPLETE")
    print("=" * 50)
    print(f"  Total processed:     {stats['total']}")
    print(f"  Agreed & inserted:   {stats['agreed_inserted']}")
    print(f"  Duplicates skipped:  {stats['duplicates_skipped']}")
    print(f"  Disagreed/skipped:   {stats['disagreed_skipped']}")
    print(f"  API errors:          {stats['api_errors']}")

    final_stats = prompts_store.get_statistics()
    print(f"\nTraining data now has {final_stats['total']} prompts")
    print(
        f"  Injections: {final_stats['injections']}"
        f" ({final_stats['injection_percentage']:.1f}%)"
    )
    print(f"  Safe: {final_stats['safe']} ({final_stats['safe_percentage']:.1f}%)")

    if not args.dry_run:
        print(f"\nClearing {unverified_path}...")
        feedback_store.clear_data()
        print("Done. Unverified submissions file cleared.")
    else:
        print("\n[DRY RUN] Would clear unverified submissions file.")


if __name__ == "__main__":
    main()
