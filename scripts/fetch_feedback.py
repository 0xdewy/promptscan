#!/usr/bin/env python3
"""Pull user feedback from the promptscan.dewy.ch server into data/v2/raw/feedback.parquet.

  python scripts/fetch_feedback.py            # rsync from apps:promptscan/api/data/, write parquet
  python scripts/fetch_feedback.py --no-sync  # convert what is already in data/feedback/

Output columns: text, label (the user's label, 1 = injection), predicted_label, confidence,
source, created_at. These labels are unverified: a user disagreeing with the model is a hint,
not ground truth. Run them through the self-play judge (scripts/selfplay.py) or review them
before adding to training.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LOCAL = ROOT / "data" / "feedback"
OUT = ROOT / "data" / "v2" / "raw" / "feedback.parquet"
REMOTE = "apps:promptscan/api/data/"


def load_rows() -> pd.DataFrame:
    frames = []
    jsonl = LOCAL / "feedback.jsonl"
    if jsonl.exists():
        rows = [json.loads(ln) for ln in jsonl.read_text(encoding="utf-8").splitlines() if ln.strip()]
        frames.append(pd.DataFrame(rows))
    legacy = LOCAL / "unverified_user_submissions.parquet"  # pre-2026-10 ParquetFeedbackStore file
    if legacy.exists():
        d = pd.read_parquet(legacy)
        d = d.rename(columns={"ensemble_confidence": "confidence", "timestamp": "created_at"})
        frames.append(d)
    if not frames:
        return pd.DataFrame(columns=["text", "user_label", "predicted_label", "confidence", "source", "created_at"])
    d = pd.concat(frames, ignore_index=True)
    for c in ("predicted_label", "confidence", "source", "created_at"):
        if c not in d:
            d[c] = None
    return d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-sync", action="store_true", help="skip rsync, convert local files only")
    a = ap.parse_args()
    LOCAL.mkdir(parents=True, exist_ok=True)
    if not a.no_sync:
        subprocess.run(["rsync", "-a", REMOTE, str(LOCAL) + "/"], check=True)

    d = load_rows()
    d["text"] = d["text"].astype(str).str.strip()
    d = d[d["text"].str.len() >= 3].drop_duplicates("text")
    d["label"] = (d["user_label"].astype(str).str.upper() == "INJECTION").astype(int)
    d["source"] = "feedback"
    d["created_at"] = d["created_at"].astype(str)  # legacy rows carry Timestamps, new rows ISO strings
    out = d[["text", "label", "predicted_label", "confidence", "source", "created_at"]]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    disagree = int((out["predicted_label"].astype(str).str.upper() != out["label"].map({1: "INJECTION", 0: "SAFE"})).sum())
    print(f"{len(out)} feedback rows -> {OUT.relative_to(ROOT)} ({disagree} disagree with the model)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
