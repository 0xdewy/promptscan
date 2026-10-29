#!/usr/bin/env python3
"""Rank original training negatives for review; never treat mining as evaluation."""

import argparse
import hashlib
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from promptscan.classifier import Classifier  # noqa: E402
from scripts.build_dataset import norm  # noqa: E402
from scripts.eval_ci import scan_scores  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, required=True, help="training negatives only"
    )
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-score", type=float, default=0.5)
    parser.add_argument("--fp32", action="store_true")
    args = parser.parse_args()
    data = pd.read_parquet(args.data)
    val = pd.read_parquet(args.validation)
    if data.empty or not data.label.eq(0).all() or not 0 <= args.min_score <= 1:
        raise ValueError("need nonempty training negatives and a valid mining cutoff")
    if set(data.text.map(norm)) & set(val.text.map(norm)):
        raise ValueError("mining pool overlaps validation")
    binary = data.text.str.contains("\x00")
    data = data[~binary].reset_index(drop=True)
    if data.empty:
        raise ValueError("mining pool contains no scannable text")
    args.out.mkdir(parents=True, exist_ok=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = Classifier.load(args.model, device=device)
    reduced = device == "cuda" and not args.fp32
    precision = (
        torch.autocast("cuda", dtype=torch.bfloat16) if reduced else nullcontext()
    )
    frames, hard = [], 0
    with (args.out / "review-candidates.jsonl").open("w") as stream, precision:
        for start in range(0, len(data), 128):
            scored = scan_scores(data.iloc[start : start + 128], classifier, batch=16)
            frames.append(scored)
            candidates = scored[scored.score >= args.min_score]
            hard += len(candidates)
            if not candidates.empty:
                stream.write(candidates.to_json(orient="records", lines=True))
            stream.flush()
            print(
                f"scored {min(start + 128, len(data))}/{len(data)}; hard {hard}",
                flush=True,
            )
    pd.concat(frames, ignore_index=True).to_parquet(
        args.out / "scores.parquet", index=False
    )
    report = {
        "training_data": str(args.data),
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "model": classifier.identity,
        "rows": len(data),
        "hard_candidates": hard,
        "binary_rows_skipped": int(binary.sum()),
        "precision": "bf16" if reduced else "fp32",
        "minimum_score": args.min_score,
        "limitation": "Ranking for label review only. Development, regression and "
        "sealed-test comparisons must use deployment fp32 scores.",
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
