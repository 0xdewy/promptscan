#!/usr/bin/env python3
"""Held-out evaluation for prompt injection detectors.

Sets: every config of Athithya117/prompt-injection-eval-suite (NotInject, OR-Bench,
XSTest, BIPIA, PIArena, LLMail, agentic pairs, ...) plus real user traffic from
lmsys/toxic-chat (test split; jailbreaking rows are positives, the rest benign).
Everything is cached under data/eval/ as parquet; build_dataset.py dedups the
training data against it.

  python scripts/eval_suite.py --scorer v2:models/release --out results/release.jsonl
  python scripts/eval_suite.py --scorer hf:deepset/deberta-v3-base-injection
  python scripts/eval_suite.py --report results/*.jsonl         # tables only

A scorer maps texts -> P(injection). The report gives per-set AUC, recall and FPR
at the detector's own threshold, and the pooled FPR at 95% recall, which is the
number that matters for "too many false positives".
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "data" / "eval"
SUITE = "Athithya117/prompt-injection-eval-suite"
SUITE_CONFIGS = [
    "notinject", "xstest", "orbench_hard", "deepset_test", "jackhhao_test",
    "simsonsun_jailbreaks", "boundary_pairs_test", "agentic5k_test", "neuralchemy_test",
    "slabs_test", "bipia", "piarena", "llmail_phase2", "synthetic_docs_test",
    "synthetic_direct_test", "synthetic_v2_test",
]
REAL_TRAFFIC = {"toxicchat"}


def load_sets() -> dict[str, pd.DataFrame]:
    """Download once, cache as parquet, return {name: df[text,label]}."""
    EVAL.mkdir(parents=True, exist_ok=True)
    from datasets import load_dataset

    sets = {}
    for cfg in SUITE_CONFIGS:
        p = EVAL / f"{cfg}.parquet"
        if not p.exists():
            ds = load_dataset(SUITE, cfg, split="test")
            pd.DataFrame({"text": ds["text"], "label": ds["label"]}).to_parquet(p)
        sets[cfg] = pd.read_parquet(p)
    p = EVAL / "toxicchat.parquet"
    if not p.exists():
        ds = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="test")
        pd.DataFrame({"text": ds["user_input"], "label": ds["jailbreaking"]}).to_parquet(p)
    sets["toxicchat"] = pd.read_parquet(p)
    for df in sets.values():
        df["text"] = df["text"].astype(str)
        df["label"] = df["label"].astype(int)
    return sets


# ---- scorers: name -> (score fn: list[str] -> np.ndarray of P(injection), threshold)

def scorer_hf(model_id: str, max_length: int = 512, batch: int = 16):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id).eval().to(dev)
    labels = {v.lower(): k for k, v in model.config.id2label.items()}
    pos = next((i for n, i in labels.items() if n in ("injection", "malicious", "jailbreak", "label_1", "1")), 1)

    @torch.no_grad()
    def score(texts):
        out = []
        for i in range(0, len(texts), batch):
            enc = tok([t[:20000] for t in texts[i:i + batch]], truncation=True, max_length=max_length,
                      padding=True, return_tensors="pt").to(dev)
            out.append(torch.softmax(model(**enc).logits.float(), -1)[:, pos].cpu().numpy())
        return np.concatenate(out)
    return score, 0.5


def scorer_v2(path: str):
    import torch

    from promptscan.classifier import Classifier
    c = Classifier.load(path, device="cuda" if torch.cuda.is_available() else "cpu")
    return (lambda texts: np.array(c.score(texts))), c.threshold


def get_scorer(name: str):
    if name.startswith("hf:"):
        return scorer_hf(name[3:])
    if name.startswith("v2:"):
        return scorer_v2(name[3:])
    raise SystemExit(f"unknown scorer {name}")


# ---- metrics

def auc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def fpr_at_recall(y: np.ndarray, s: np.ndarray, recall: float = 0.95) -> tuple[float, float]:
    """Threshold that catches `recall` of positives, and the FPR there."""
    pos = np.sort(s[y == 1])
    thr = pos[int((1 - recall) * len(pos))]
    return float((s[y == 0] >= thr).mean()), float(thr)


def report(files: list[str]) -> str:
    rows = []
    for f in files:
        df = pd.read_json(f, lines=True)
        name = df["scorer"].iloc[0]
        thr = float(df["threshold"].iloc[0])
        y, s = df["label"].to_numpy(), df["score"].to_numpy()
        per = []
        for set_name, g in df.groupby("set"):
            yy, ss = g["label"].to_numpy(), g["score"].to_numpy()
            pred = ss >= thr
            per.append(dict(
                set=set_name, n=len(g), pos=int(yy.sum()),
                auc=auc(yy, ss),
                recall=float(pred[yy == 1].mean()) if yy.sum() else float("nan"),
                fpr=float(pred[yy == 0].mean()) if (yy == 0).sum() else float("nan"),
            ))
        fpr95, thr95 = fpr_at_recall(y, s)
        real = df[df["set"].isin(REAL_TRAFFIC)]
        rows.append((name, thr, per, dict(
            auc=auc(y, s), recall=float((s >= thr)[y == 1].mean()), fpr=float((s >= thr)[y == 0].mean()),
            fpr95=fpr95, thr95=thr95,
            real_fpr=float((real["score"].to_numpy() >= thr)[real["label"].to_numpy() == 0].mean()) if len(real) else float("nan"),
            ms=float(df["ms"].mean()),
        )))
    out = ["| scorer | thr | AUC | recall@thr | FPR@thr | FPR@95% recall | real-traffic FPR | ms/prompt |", "|---|---|---|---|---|---|---|---|"]
    for name, thr, _, m in rows:
        out.append(f"| {name} | {thr:.2f} | {m['auc']:.3f} | {m['recall']:.3f} | {m['fpr']:.3f} | {m['fpr95']:.3f} (thr {m['thr95']:.2f}) | {m['real_fpr']:.3f} | {m['ms']:.0f} |")
    out.append("")
    out.append("Per set (recall / FPR at the detector's threshold):")
    out.append("")
    sets = sorted({p["set"] for r in rows for p in r[2]})
    out.append("| set | n | pos | " + " | ".join(r[0] for r in rows) + " |")
    out.append("|---|---|---|" + "---|" * len(rows))
    for set_name in sets:
        cells, p0 = [], None
        for _, _, per, _ in rows:
            p = next((x for x in per if x["set"] == set_name), None)
            if p is None:
                cells.append("-")
                continue
            p0 = p0 or p
            cells.append(f"{p['recall']:.2f} / {p['fpr']:.2f}" if p["pos"] and p["n"] - p["pos"] else
                         (f"rec {p['recall']:.2f}" if p["pos"] else f"fpr {p['fpr']:.2f}"))
        out.append(f"| {set_name} | {p0['n']} | {p0['pos']} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scorer")
    ap.add_argument("--out")
    ap.add_argument("--limit", type=int, default=0, help="cap rows per set (smoke test)")
    ap.add_argument("--report", nargs="*")
    a = ap.parse_args()
    if a.report:
        print(report(a.report))
        return 0
    sets = load_sets()
    score, thr = get_scorer(a.scorer)
    out = Path(a.out or f"results/{a.scorer.replace('/', '_').replace(':', '_')}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for name, df in sets.items():
            if a.limit:
                df = df.sample(min(a.limit, len(df)), random_state=0)
            t0 = time.perf_counter()
            s = score(df["text"].tolist())
            ms = 1000 * (time.perf_counter() - t0) / len(df)
            for text, label, sc in zip(df["text"], df["label"], s):
                f.write(json.dumps({"scorer": a.scorer, "threshold": thr, "set": name, "label": int(label),
                                    "score": float(sc), "ms": ms, "text": text[:200]}) + "\n")
            yy = df["label"].to_numpy()
            pred = s >= thr
            rec = pred[yy == 1].mean() if yy.sum() else float("nan")
            fpr = pred[yy == 0].mean() if (yy == 0).sum() else float("nan")
            print(f"{name:24s} n={len(df):5d} recall={rec:.3f} fpr={fpr:.3f} {ms:.0f}ms", file=sys.stderr, flush=True)
    print(report([str(out)]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
