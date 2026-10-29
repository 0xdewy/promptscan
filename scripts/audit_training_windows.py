#!/usr/bin/env python3
"""Find encoded views eligible for supervision under opposing document labels.

This is an ambiguity audit, not proof that a document-level max loss is infeasible.
No model inference or held-out data is needed.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from promptscan.scan import SCANNER_POLICY_VERSION  # noqa: E402
from scripts.context_baseline import legacy_segments  # noqa: E402
from scripts.train_v2 import encode_documents  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", default="models/release")
    parser.add_argument("--context-tokens", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--legacy", action="store_true", help="audit context-1 inputs")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.legacy and args.context_tokens:
        parser.error("--legacy requires --context-tokens 0")
    data = pd.read_parquet(args.data)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    examples = {}
    windows = 0
    for start in range(0, len(data), 64):
        chunk = data.iloc[start : start + 64]
        encoded = encode_documents(
            chunk.to_dict("list"),
            tokenizer,
            args.max_length,
            args.context_tokens,
            segmenter=legacy_segments if args.legacy else None,
        )
        for owner, row in enumerate(chunk.itertuples()):
            for ids, eligible in zip(
                encoded["input_ids"][owner],
                encoded["attack_windows"][owner],
                strict=True,
            ):
                if row.label and not eligible:
                    continue
                windows += 1
                key = hashlib.sha256(np.asarray(ids, dtype="<i4").tobytes()).hexdigest()
                examples.setdefault(key, {}).setdefault(int(row.label), start + owner)
    conflicts = [
        {"window_sha256": key, "negative_row": rows[0], "positive_row": rows[1]}
        for key, rows in sorted(examples.items())
        if len(rows) > 1
    ]
    args.out.mkdir(parents=True, exist_ok=False)
    report = {
        "data": str(args.data),
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "rows": len(data),
        "eligible_windows": windows,
        "unique_windows": len(examples),
        "opposite_label_eligible_windows": len(conflicts),
        "context_tokens": args.context_tokens,
        "max_length": args.max_length,
        "scanner_policy": "context-1" if args.legacy else SCANNER_POLICY_VERSION,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.out / "conflicts.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in conflicts)
    )
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
