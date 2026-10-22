#!/usr/bin/env python3
"""Benchmark promptscan's ensemble against deepset/deberta-v3-base-injection.

Datasets (Hugging Face):
  deepset/prompt-injections (test)        in-distribution: both models trained on it
  xTRam1/safe-guard-prompt-injection (test) held out for promptscan
  jackhhao/jailbreak-classification (test)  held out, jailbreaks rather than injections

Usage: python scripts/benchmark.py [--samples N] [--skip-deberta] [--out docs/BENCHMARK.md]
"""
import argparse, random, time, sys
from datetime import date

def load(name, split, text_key, label_fn, n, seed=0):
    from datasets import load_dataset
    ds = load_dataset(name, split=split)
    rows = [(r[text_key], label_fn(r)) for r in ds if r[text_key] and str(r[text_key]).strip()]
    random.Random(seed).shuffle(rows)
    return rows[:n] if n else rows

SETS = [
    ("deepset/prompt-injections", "test", "text", lambda r: int(r["label"]) == 1),
    ("xTRam1/safe-guard-prompt-injection", "test", "text", lambda r: int(r["label"]) == 1),
    ("jackhhao/jailbreak-classification", "test", "prompt", lambda r: str(r["type"]).lower() == "jailbreak"),
]

def metrics(preds, labels, times):
    tp = sum(p and l for p, l in zip(preds, labels)); fp = sum(p and not l for p, l in zip(preds, labels))
    fn = sum((not p) and l for p, l in zip(preds, labels)); tn = len(labels) - tp - fp - fn
    prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(n=len(labels), acc=(tp + tn) / len(labels), prec=prec, rec=rec, f1=f1, ms=1000 * sum(times) / len(times))

def run_promptscan(rows):
    from promptscan import UnifiedDetector
    d = UnifiedDetector(model_type="ensemble")
    preds, times = [], []
    for text, _ in rows:
        t = time.perf_counter(); r = d.predict(text[:10000]); times.append(time.perf_counter() - t)
        preds.append(r["prediction"] == "INJECTION")
    return preds, times

def run_deberta(rows):
    from transformers import pipeline
    clf = pipeline("text-classification", model="deepset/deberta-v3-base-injection", truncation=True, max_length=512, device=-1)
    preds, times = [], []
    for text, _ in rows:
        t = time.perf_counter(); r = clf(text[:4000])[0]; times.append(time.perf_counter() - t)
        preds.append(r["label"].upper() == "INJECTION")
    return preds, times

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--skip-deberta", action="store_true"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    lines = [f"# Benchmark ({date.today()})", "", f"Command: `python scripts/benchmark.py --samples {a.samples}`. CPU only, batch size 1, one sample at a time.", "",
             "| dataset | model | n | accuracy | precision | recall | F1 | ms/sample |", "|---|---|---|---|---|---|---|---|"]
    for name, split, key, lab, in SETS:
        rows = load(name, split, key, lab, a.samples); labels = [l for _, l in rows]
        runners = [("promptscan ensemble", run_promptscan)] + ([] if a.skip_deberta else [("deepset/deberta-v3-base-injection", run_deberta)])
        for mname, fn in runners:
            preds, times = fn(rows); m = metrics(preds, labels, times)
            lines.append(f"| {name} ({split}) | {mname} | {m['n']} | {m['acc']:.3f} | {m['prec']:.3f} | {m['rec']:.3f} | {m['f1']:.3f} | {m['ms']:.1f} |")
            print(lines[-1], file=sys.stderr, flush=True)
    lines += ["", "Notes:", "- deepset/prompt-injections is in promptscan's training data and is what the deberta model was fine-tuned on, so that row is in-distribution for both.",
              "- The other two sets are not in promptscan's training data. jailbreak-classification labels jailbreaks, a neighbouring task.",
              "- Latency is per-sample wall time on a 2-vCPU Hetzner box, tokenization included, no batching."]
    out = "\n".join(lines) + "\n"
    print(out)
    if a.out: open(a.out, "w").write(out)

if __name__ == "__main__":
    main()
