#!/usr/bin/env python3
"""Accept only a fixed-threshold validation improvement. Exit 1 rejects, 2 errors."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train_v2 import HARD  # noqa: E402


def measure(data: pd.DataFrame, scores, threshold: float) -> dict:
    if len(data) != len(scores) or not len(data):
        raise ValueError("validation needs one score per row and a nonempty dataset")
    scores = pd.Series(scores, index=data.index)
    if not scores.between(0, 1).all() or not data.label.isin([0, 1]).all():
        raise ValueError("invalid scores or labels in validation")
    predictions = scores >= threshold
    positive = data.label == 1
    hard = ~positive & data.source.isin(HARD)
    if not positive.any() or positive.all():
        raise ValueError("validation needs positive and negative examples")
    return {
        "threshold": threshold,
        "rows": len(data),
        "positives": int(positive.sum()),
        "negatives": int((~positive).sum()),
        "hard_negatives": int(hard.sum()),
        "tp": int(predictions[positive].sum()),
        "fp": int(predictions[~positive].sum()),
        "hard_fp": int(predictions[hard].sum()),
    }


def compare(baseline: dict, candidate: dict) -> list[str]:
    """Return rejection reasons; counts avoid rounding small regressions away."""
    for name in ("threshold", "rows", "positives", "negatives", "hard_negatives"):
        if baseline[name] != candidate[name]:
            raise ValueError(f"validation {name} differs between models")
    reasons = []
    minimized = ["fp", "hard_fp"]
    if "real_traffic_fp" in baseline:
        minimized.append("real_traffic_fp")
    for name in minimized:
        if candidate[name] > baseline[name]:
            reasons.append(f"{name} increased: {baseline[name]} -> {candidate[name]}")
    if candidate["tp"] < baseline["tp"]:
        reasons.append(f"tp decreased: {baseline['tp']} -> {candidate['tp']}")
    if all(candidate[k] == baseline[k] for k in ["tp", *minimized]):
        reasons.append("no validation improvement")
    return reasons


def compare_score_files(before: pd.DataFrame, after: pd.DataFrame):
    columns = ["set", "text", "label"]
    if not before[columns].equals(after[columns]):
        raise ValueError("benchmark populations differ")
    thresholds = before.threshold.unique()
    if len(thresholds) != 1 or not (after.threshold == thresholds[0]).all():
        raise ValueError("benchmark thresholds differ")
    measured = []
    for data in (before, after):
        counts = measure(data.assign(source=""), data.score, float(thresholds[0]))
        real = (data.label == 0) & (data["set"] == "toxicchat")
        counts["real_traffic_fp"] = int(
            (data.loc[real, "score"] >= thresholds[0]).sum()
        )
        measured.append(counts)
    return measured[0], measured[1], compare(*measured)


def main() -> int:
    import torch

    from promptscan.classifier import Classifier

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--data", type=Path, help="fixed validation parquet")
    ap.add_argument(
        "--scores",
        action="store_true",
        help="compare baseline/candidate benchmark JSONL files instead",
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if not args.scores and args.data is None:
        ap.error("model validation requires --data")
    try:
        if args.scores:
            before, after, reasons = compare_score_files(
                pd.read_json(args.baseline, lines=True),
                pd.read_json(args.candidate, lines=True),
            )
            result = {
                "accepted": not reasons,
                "reasons": reasons,
                "baseline": before,
                "candidate": after,
            }
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, indent=2))
            return 1 if reasons else 0
        data = pd.read_parquet(args.data)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        baseline = Classifier.load(args.baseline, device=device)
        threshold = baseline.threshold
        before = measure(data, baseline.score(data.text.tolist(), batch=32), threshold)
        del baseline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        candidate = Classifier.load(args.candidate, device=device)
        after = measure(data, candidate.score(data.text.tolist(), batch=32), threshold)
        reasons = compare(before, after)
        if candidate.threshold != threshold:
            reasons.append("candidate's stored threshold differs from baseline")
        result = {
            "accepted": not reasons,
            "reasons": reasons,
            "baseline_model": args.baseline,
            "candidate_model": args.candidate,
            "validation_data": str(args.data),
            "baseline": before,
            "candidate": after,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return 1 if reasons else 0
    except torch.OutOfMemoryError:
        raise  # preserve the CUDA error type so the driver can retry it once
    except (OSError, ValueError, RuntimeError, KeyError, AttributeError) as error:
        print(f"validation error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
