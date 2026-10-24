#!/usr/bin/env python3
"""Fine-tune a DeBERTa-v3 classifier on data/v2/{train,val}.parquet (text, label).

  python scripts/train_v2.py --base microsoft/deberta-v3-xsmall --out models/v2-xsmall
  python scripts/train_v2.py --base microsoft/deberta-v3-small --out models/v2-small --lr 2e-5

Keeps the last epoch, picks the decision threshold on val
(<=1% FPR on hard benign rows), and stores it in the model config so Classifier.load() uses it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from promptscan.classifier import Classifier  # noqa: E402

# Benign sources that look like attacks (role-play, system prompts, near-miss pairs, edgy requests).
# Easy negatives (wiki, recipes) say nothing about where the threshold should sit.
HARD = {"agentic", "boundary", "neuralchemy", "slabs", "jackhhao", "deepset", "inthewild", "awesome",
        "spml", "orbench", "xtram"}


def best_threshold(p_neg: np.ndarray, fpr: float = 0.01) -> float:
    """Lowest threshold whose FPR on hard val negatives is <= fpr. False positives cost more than misses."""
    return float(max(np.quantile(p_neg, 1 - fpr), 0.5))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="microsoft/deberta-v3-xsmall")
    ap.add_argument("--data", default="data/v2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4, help="gradient accumulation steps")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="train rows cap (smoke test)")
    ap.add_argument("--drop-sources", nargs="*", default=[], help="source prefixes to leave out (ablations)")
    ap.add_argument("--eval-steps", type=int, default=200, help="evaluate val + FP probe every N optimizer steps")
    a = ap.parse_args()

    train = pd.read_parquet(f"{a.data}/train.parquet")
    val = pd.read_parquet(f"{a.data}/val.parquet")
    if a.drop_sources:
        train = train[~train.source.str.startswith(tuple(a.drop_sources))]
        val = val[~val.source.str.startswith(tuple(a.drop_sources))]
    if a.limit:   # subsample, but always keep the (small, targeted) self-play rows
        sp = train.source.str.startswith(("selfplay", "rl_"))
        train = pd.concat([train[sp], train[~sp].sample(a.limit - sp.sum(), random_state=0)])
    print(f"train {len(train)} ({train.label.mean():.2%} pos), val {len(val)} ({val.label.mean():.2%} pos)")

    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModelForSequenceClassification.from_pretrained(   # checkpoints are fp16; train in fp32 weights
        a.base, num_labels=2, dtype=torch.float32, id2label={0: "SAFE", 1: "INJECTION"}, label2id={"SAFE": 0, "INJECTION": 1})

    def enc(b):
        return tok(b["text"], truncation=True, max_length=a.max_length)

    cols = ["text", "label"]
    dtrain = Dataset.from_pandas(train[cols], preserve_index=False).map(enc, batched=True, remove_columns=["text"])
    dval = Dataset.from_pandas(val[cols], preserve_index=False).map(enc, batched=True, remove_columns=["text"])
    # Watch-only FP probe: benign real traffic + benign texts full of trigger words (never used to pick anything).
    probe = pd.concat([pd.read_parquet(f"data/eval/{n}.parquet") for n in ("toxicchat", "notinject")])
    probe = probe[probe.label == 0].sample(600, random_state=0)
    dprobe = Dataset.from_pandas(probe[cols], preserve_index=False).map(enc, batched=True, remove_columns=["text"])

    def metrics(ev):
        logits, y = ev
        p = torch.softmax(torch.tensor(logits).float(), -1)[:, 1].numpy()
        m = {"fpr50": float((p[y == 0] >= 0.5).mean())}
        if 0 < y.sum() < len(y):
            m |= {"auc": roc_auc_score(y, p), "recall50": float((p[y == 1] >= 0.5).mean())}
        return m

    args = TrainingArguments(
        output_dir=f"/data/codex-scratch/user/codex-job-manual-promptscan/{Path(a.out).name}-ckpt", num_train_epochs=a.epochs, learning_rate=a.lr,
        per_device_train_batch_size=a.batch, gradient_accumulation_steps=a.accum,
        per_device_eval_batch_size=32, train_sampling_strategy="group_by_length",
        warmup_ratio=0.06, weight_decay=0.01, lr_scheduler_type="cosine",
        bf16=torch.cuda.is_available(),
        eval_strategy="steps", eval_steps=a.eval_steps, save_strategy="no",
        logging_steps=50, report_to=[], dataloader_num_workers=4, seed=0,
    )
    trainer = Trainer(model=model, args=args, train_dataset=dtrain, eval_dataset={"val": dval, "probe": dprobe},
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics)
    trainer.train()

    # score val exactly as inference does (windows, max), in row order
    clf = Classifier(trainer.model, tok, 0.5, a.max_length, device=trainer.model.device)
    p = np.array(clf.score(val["text"].tolist(), batch=32))
    y = val["label"].to_numpy()
    hard = (y == 0) & val["source"].isin(HARD).to_numpy()
    clf.threshold = thr = best_threshold(p[hard])
    final = {"auc": float(roc_auc_score(y, p)), "threshold": thr, "hard_negatives": int(hard.sum()),
             "recall": float((p >= thr)[y == 1].mean()), "fpr": float((p >= thr)[y == 0].mean()),
             "hard_fpr": float((p >= thr)[hard].mean())}
    print(json.dumps(final, indent=1))
    clf.save(
        a.out, base=a.base, train_rows=len(train), epochs=a.epochs, lr=a.lr, val=final)
    print(f"saved {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
