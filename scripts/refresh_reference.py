#!/usr/bin/env python3
"""Refresh replay floors from correct teacher predictions on training rows only."""

import argparse
import gc
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from promptscan.classifier import Classifier  # noqa: E402
from scripts.train_v2 import document_logits, encode_documents  # noqa: E402


@torch.inference_mode()
def eligible_margins(data, classifier, batch=8):
    """Measure the exact supervised region, excluding known-clean attack context."""
    values = []
    for start in range(0, len(data), 64):
        chunk = data.iloc[start : start + 64]
        encoded = encode_documents(
            chunk.to_dict("list"),
            classifier.tokenizer,
            classifier.max_length,
            classifier.context_tokens,
        )
        windows, owners, allowed = [], [], []
        for owner in range(len(chunk)):
            for ids, mask, eligible in zip(
                encoded["input_ids"][owner],
                encoded["attention_mask"][owner],
                encoded["attack_windows"][owner],
                strict=True,
            ):
                windows.append({"input_ids": ids, "attention_mask": mask})
                owners.append(owner)
                allowed.append(bool(eligible or chunk.label.iloc[owner] == 0))
        order = sorted(range(len(windows)), key=lambda i: len(windows[i]["input_ids"]))
        logits = torch.empty((len(windows), 2))
        for first in range(0, len(order), batch):
            indices = order[first : first + batch]
            inputs = classifier.tokenizer.pad(
                [windows[i] for i in indices], return_tensors="pt"
            ).to(classifier.device)
            output = classifier.model(**inputs).logits.float().cpu()
            if classifier.pos == 0:
                output = output[:, [1, 0]]
            logits[indices] = output
        pooled = document_logits(
            logits,
            torch.tensor(owners),
            len(chunk),
            torch.tensor(allowed),
        )
        values.extend(pooled[:, 1].tolist())
        print(f"scored {min(start + 64, len(data))}/{len(data)}", flush=True)
    return np.asarray(values)


def merge_correct_references(labels, existing, margins):
    if not np.isfinite(margins).all():
        raise ValueError("teacher margins must be finite")
    direction = np.asarray(labels, dtype=np.int64) * 2 - 1
    strength = np.asarray(existing, dtype=float) * direction
    strength = np.where(np.isfinite(strength) & (strength > 0), strength, -np.inf)
    current = np.asarray(margins) * direction
    strength = np.maximum(strength, np.where(current > 0, current, -np.inf))
    return np.where(np.isfinite(strength), strength * direction, np.nan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--teacher", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    train = pd.read_parquet(args.data / "train.parquet")
    original = train[["text", "label"]].copy()
    references = train.get("reference_margin", pd.Series(np.nan, index=train.index))
    args.out.mkdir(parents=True, exist_ok=False)
    report = {
        "input": str(args.data),
        "rows": len(train),
        "input_sha256": hashlib.sha256(
            (args.data / "train.parquet").read_bytes()
        ).hexdigest(),
        "teacher_inputs": "Complete scanner views and eligible payload windows; fp32",
        "labels_or_membership_changed": False,
        "teachers": [],
    }
    for i, path in enumerate(args.teacher):
        teacher = Classifier.load(
            path, device="cuda" if torch.cuda.is_available() else "cpu"
        )
        if teacher.context_tokens != 0 or teacher.max_length != 512:
            raise ValueError("reference refresh requires ordinary 512-token windows")
        margins = eligible_margins(train, teacher)
        references = merge_correct_references(train.label, references, margins)
        pd.DataFrame({"margin": margins}).to_parquet(args.out / f"teacher-{i}.parquet")
        report["teachers"].append(
            {
                "model": teacher.identity,
                "correct_training_rows": int(
                    ((train.label.to_numpy(dtype=int) * 2 - 1) * margins > 0).sum()
                ),
            }
        )
        del teacher
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        (args.out / "reference.json").write_text(json.dumps(report, indent=2) + "\n")
    train["reference_margin"] = references
    assert train[["text", "label"]].equals(original)
    train.to_parquet(args.out / "train.parquet", index=False)
    shutil.copyfile(args.data / "val.parquet", args.out / "val.parquet")
    report["anchored_rows"] = int(np.isfinite(references).sum())
    report["output_sha256"] = hashlib.sha256(
        (args.out / "train.parquet").read_bytes()
    ).hexdigest()
    (args.out / "reference.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
