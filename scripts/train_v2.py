#!/usr/bin/env python3
"""Fine-tune a classifier on complete documents in {train,val}.parquet.

  python scripts/train_v2.py --base microsoft/deberta-v3-xsmall --out models/v2-xsmall
  python scripts/train_v2.py --base models/release --out models/candidate

Keeps the last epoch, picks the decision threshold on val
(<= --fpr-target on hard benign rows), and stores it in the model config.
Pass --threshold to preserve a fixed decision threshold during an experiment.
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
from torch.utils.checkpoint import checkpoint
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from promptscan.classifier import Classifier  # noqa: E402
from promptscan.scan import (  # noqa: E402
    SCANNER_POLICY_VERSION,  # noqa: E402
    classifier_segments,
    hidden_text,
)
from promptscan.windows import encode_windows  # noqa: E402

# Benign sources that look like attacks: instructions, role-play and near misses.
# Easy negatives (wiki, recipes) say nothing about where the threshold should sit.
HARD = {
    "agentic",
    "boundary",
    "neuralchemy",
    "slabs",
    "jackhhao",
    "deepset",
    "inthewild",
    "awesome",
    "spml",
    "orbench",
    "xtram",
    "ci_repository",
}


def best_threshold(p_neg: np.ndarray, fpr: float = 0.01) -> float:
    """Smallest cutoff within an exact negative error budget, including ties."""
    p_neg = np.asarray(p_neg, dtype=float)
    if not len(p_neg) or not np.isfinite(p_neg).all():
        raise ValueError("calibration needs finite negative scores")
    if not 0 <= fpr < 1 or not ((0 <= p_neg) & (p_neg <= 1)).all():
        raise ValueError("invalid false-positive target or scores")
    ordered = np.sort(p_neg)[::-1]
    cutoff = float(np.nextafter(ordered[int(len(ordered) * fpr)], np.inf))
    if cutoff > 1:
        raise ValueError("no cutoff <= 1 meets the false-positive budget")
    return cutoff


def encode_documents(batch, tokenizer, max_length, context_tokens=0, segmenter=None):
    """Keep every scanner segment and window, grouped by its document label."""
    segmenter = segmenter or classifier_segments
    segments, owners, origins, spans = [], [], [], []
    for owner, text in enumerate(batch["text"]):
        start = batch.get("attack_start", [None] * len(batch["text"]))[owner]
        end = batch.get("attack_end", [None] * len(batch["text"]))[owner]
        span = None
        if start is not None and end is not None and pd.notna(start) and pd.notna(end):
            start, end = int(start), int(end)
            if not 0 <= start < end <= len(text):
                raise ValueError("attack span lies outside its document")
            span = (start, end)
        spans.append(span)
        for segment, _, where in segmenter(text, hidden_text(text)):
            segments.append(segment)
            owners.append(owner)
            origin = text.find(segment) if where else 0
            if origin < 0 and len(segment) == len(text):
                origin = 0  # tag decoding retained a one-to-one character map
            origins.append(origin if origin >= 0 else None)
    encoded = encode_windows(
        tokenizer, segments, max_length, context_tokens, offsets=True
    )
    grouped = {
        key: [[] for _ in batch["text"]] for key in ("input_ids", "attention_mask")
    }
    grouped["attack_windows"] = [[] for _ in batch["text"]]
    for window, segment in enumerate(encoded["overflow_to_sample_mapping"]):
        document = owners[segment]
        for key in ("input_ids", "attention_mask"):
            grouped[key][document].append(encoded[key][window])
        span, origin = spans[document], origins[segment]
        allowed = (
            span is None
            or origin is None
            or any(
                right > left and origin + left < span[1] and origin + right > span[0]
                for left, right in encoded["offset_mapping"][window]
            )
        )
        grouped["attack_windows"][document].append(allowed)
    for owner, span in enumerate(spans):
        if span is not None and not any(grouped["attack_windows"][owner]):
            raise ValueError("attack span is outside the visible training windows")
    return grouped


class DocumentCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        windows, owners, attack_windows = [], [], []
        for owner, feature in enumerate(features):
            allowed = feature.get("attack_windows", [True] * len(feature["input_ids"]))
            for ids, mask, attack in zip(
                feature["input_ids"], feature["attention_mask"], allowed, strict=True
            ):
                windows.append({"input_ids": ids, "attention_mask": mask})
                owners.append(owner)
                attack_windows.append(attack)
        batch = self.tokenizer.pad(windows, return_tensors="pt")
        batch["document"] = torch.tensor(owners)
        batch["labels"] = torch.tensor([f["label"] for f in features])
        batch["attack_windows"] = torch.tensor(attack_windows, dtype=torch.bool)
        batch["reference_margin"] = torch.tensor(
            [
                float(f["reference_margin"])
                if f.get("reference_margin") is not None
                else float("nan")
                for f in features
            ]
        )
        return batch


def document_logits(logits, owners, count, allowed=None):
    """Max injection probability per document, exactly as Classifier.score does."""
    margins = logits.float()[:, 1] - logits.float()[:, 0]
    pooled = torch.stack(
        [
            margins[
                (owners == i) if allowed is None else ((owners == i) & allowed)
            ].max()
            for i in range(count)
        ]
    )
    return torch.stack((torch.zeros_like(pooled), pooled), dim=-1)


def document_loss(
    logits,
    owners,
    labels,
    attack_windows=None,
    reference_margin=None,
    anchor_weight=0.0,
    distill=False,
):
    pooled = document_logits(logits, owners, len(labels))
    if attack_windows is None:
        attack_windows = torch.ones_like(owners, dtype=torch.bool)
    clean = (labels[owners] == 0) | ~attack_windows
    selected = attack_windows | (labels[owners] == 0)
    target_logits = document_logits(logits, owners, len(labels), selected)
    target_loss = torch.nn.functional.cross_entropy(
        target_logits, labels, reduction="none"
    )
    # Correct all known-clean windows as well as the worst one. An annotated
    # attack's clean context is negative; its winning payload window is positive.
    window_loss = torch.nn.functional.softplus(
        logits.float()[:, 1] - logits.float()[:, 0]
    )
    totals = window_loss.new_zeros(len(labels)).scatter_add(
        0, owners, window_loss * clean
    )
    counts = torch.bincount(owners[clean], minlength=len(labels)).clamp_min(1)
    clean_loss = totals / counts
    # Max and mean are two views of the same negative supervision, not two
    # separate documents. A one-window document must retain ordinary CE weight.
    combined = target_loss + clean_loss
    loss = torch.where(labels == 0, combined / 2, combined).mean()
    if reference_margin is not None and anchor_weight:
        known = torch.isfinite(reference_margin)
        direction = labels * 2 - 1
        if distill:
            # Preserve confident, correct teacher decisions. Weak or wrong
            # teacher targets leave the ground-truth loss free to improve them.
            known &= reference_margin * direction >= 4.5
        if known.any():
            if distill:
                temperature = 2.0
                reference = reference_margin[known].clamp(-12, 12)
                teacher = torch.stack((torch.zeros_like(reference), reference), -1)
                penalty = (
                    torch.nn.functional.kl_div(
                        (target_logits[known] / temperature).log_softmax(-1),
                        (teacher / temperature).softmax(-1),
                        reduction="sum",
                    )
                    * temperature**2
                )
                return loss + anchor_weight * penalty / len(labels), pooled
            # Preserve useful decision margins with a bounded target.
            # Contextual negatives have no teacher target, so old false alarms
            # on those examples are never reinforced by this term.
            margin = target_logits[known, 1] - target_logits[known, 0]
            direction = labels[known] * 2 - 1
            reference = reference_margin[known].clamp(-6, 6)
            degradation = ((reference - margin) * direction).relu()
            loss = loss + anchor_weight * torch.nn.functional.smooth_l1_loss(
                degradation, torch.zeros_like(degradation), reduction="sum"
            ) / len(labels)
    return loss, pooled


def forward_windows(model, inputs, window_batch=8):
    """Retain every window while bounding activation memory per forward pass."""
    count = len(inputs["input_ids"])
    if count <= window_batch:
        return model(**inputs).logits

    def forward(ids, mask):
        return model(input_ids=ids, attention_mask=mask).logits

    chunks = []
    for start in range(0, count, window_batch):
        ids = inputs["input_ids"][start : start + window_batch]
        mask = inputs["attention_mask"][start : start + window_batch]
        chunks.append(
            checkpoint(forward, ids, mask, use_reentrant=False)
            if model.training and torch.is_grad_enabled()
            else forward(ids, mask)
        )
    return torch.cat(chunks)


class DocumentTrainer(Trainer):
    def __init__(
        self, *args, anchor_weight=0.0, window_batch=8, distill=False, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.anchor_weight = anchor_weight
        self.window_batch = window_batch
        self.distill = distill
        self.model_accepts_loss_kwargs = False  # loss is a mean over documents

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs.pop("labels")
        owners = inputs.pop("document")
        attack_windows = inputs.pop("attack_windows", None)
        reference_margin = inputs.pop("reference_margin", None)
        loss, logits = document_loss(
            forward_windows(model, inputs, self.window_batch),
            owners,
            labels,
            attack_windows,
            reference_margin,
            self.anchor_weight,
            self.distill,
        )
        return (loss, {"logits": logits}) if return_outputs else loss


def train_last_layers(model, count):
    """Refine upper encoder layers and the classifier, preserving lower features."""
    encoder = getattr(model.base_model, "encoder", None)
    layers = getattr(encoder, "layer", None)
    if layers is None or not 1 <= count <= len(layers):
        raise ValueError("train-last-layers needs an encoder and a valid layer count")
    model.base_model.requires_grad_(False)
    for layer in layers[-count:]:
        layer.requires_grad_(True)
    pooler = getattr(model.base_model, "pooler", None)
    if pooler is not None:
        pooler.requires_grad_(True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="microsoft/deberta-v3-xsmall")
    ap.add_argument("--data", default="data/v2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--scores-out",
        type=Path,
        help="save validation scanner scores for cutoff comparisons",
    )
    ap.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="optional optimizer-step cap for a short comparison",
    )
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument(
        "--train-last-layers",
        type=int,
        help="optional upper encoder layer count; freezes embeddings and lower layers",
    )
    ap.add_argument(
        "--window-batch",
        type=int,
        default=8,
        help="maximum windows per forward pass; all document windows remain supervised",
    )
    ap.add_argument(
        "--anchor-weight",
        type=float,
        default=0.0,
        help="optional replay-margin preservation weight; needs reference_margin data",
    )
    ap.add_argument(
        "--distill",
        action="store_true",
        help="preserve confident correct teacher distributions, not a margin floor",
    )
    ap.add_argument(
        "--save-steps",
        type=int,
        default=0,
        help="optional interval for weights-only checkpoints selected on"
        " development data",
    )
    ap.add_argument(
        "--fpr-target",
        type=float,
        default=0.01,
        help="negative calibration error budget when choosing a cutoff",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        help="keep a fixed decision threshold instead of recalibrating",
    )
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4, help="gradient accumulation steps")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument(
        "--context-tokens",
        type=int,
        default=0,
        help="carry this many document prefix/footer tokens into long windows",
    )
    ap.add_argument("--limit", type=int, default=0, help="train rows cap (smoke test)")
    ap.add_argument(
        "--drop-sources",
        nargs="*",
        default=[],
        help="source prefixes to leave out (ablations)",
    )
    ap.add_argument(
        "--eval-steps",
        type=int,
        default=200,
        help="evaluate validation documents every N steps; 0 disables interim eval",
    )
    a = ap.parse_args()
    if a.window_batch < 1:
        ap.error("--window-batch must be positive")
    if a.threshold is not None and not 0 < a.threshold <= 1:
        ap.error("--threshold must be in (0, 1]")
    if not 0 <= a.fpr_target < 1:
        ap.error("--fpr-target must be in [0, 1)")
    if not 0 <= a.context_tokens <= a.max_length // 4:
        ap.error("--context-tokens must be between zero and max-length / 4")
    if not np.isfinite(a.anchor_weight) or a.anchor_weight < 0:
        ap.error("--anchor-weight must be finite and nonnegative")
    if a.distill and not a.anchor_weight:
        ap.error("--distill requires a positive --anchor-weight")

    train = pd.read_parquet(f"{a.data}/train.parquet")
    val = pd.read_parquet(f"{a.data}/val.parquet")
    if a.anchor_weight and (
        "reference_margin" not in train or not train.reference_margin.notna().any()
    ):
        ap.error("replay anchoring requires finite reference_margin targets")
    if a.drop_sources:
        train = train[~train.source.str.startswith(tuple(a.drop_sources))]
        val = val[~val.source.str.startswith(tuple(a.drop_sources))]
    if a.limit and a.limit < len(train):
        sp = train.source.str.startswith(("selfplay", "rl_"))
        if int(sp.sum()) > a.limit:
            ap.error("--limit is smaller than the targeted training rows")
        train = pd.concat(
            [train[sp], train[~sp].sample(a.limit - sp.sum(), random_state=0)]
        )
    if "repo" in train and "repo" in val:
        train_repos = set(train.repo.dropna().str.casefold())
        val_repos = set(val.repo.dropna().str.casefold())
        if train_repos & val_repos:
            raise ValueError("training and validation repositories overlap")
    print(
        f"train {len(train)} ({train.label.mean():.2%} pos), "
        f"val {len(val)} ({val.label.mean():.2%} pos)"
    )

    tok = AutoTokenizer.from_pretrained(a.base)
    # Checkpoints may be fp16; train with fp32 weights.
    model = AutoModelForSequenceClassification.from_pretrained(
        a.base,
        num_labels=2,
        dtype=torch.float32,
        use_safetensors=True,
        id2label={0: "SAFE", 1: "INJECTION"},
        label2id={"SAFE": 0, "INJECTION": 1},
    )
    if a.train_last_layers is not None:
        train_last_layers(model, a.train_last_layers)
    model.config.promptscan = {
        "context_tokens": a.context_tokens,
        "training_context_tokens": a.context_tokens,
        "max_length": a.max_length,
        "threshold": a.threshold if a.threshold is not None else 0.5,
        "scanner_policy": SCANNER_POLICY_VERSION,
    }

    def encode_frame(frame):
        columns = [
            c
            for c in ("text", "label", "attack_start", "attack_end", "reference_margin")
            if c in frame
        ]
        return Dataset.from_pandas(frame[columns], preserve_index=False).map(
            lambda b: encode_documents(b, tok, a.max_length, a.context_tokens),
            batched=True,
            remove_columns=[
                c for c in columns if c not in ("label", "reference_margin")
            ],
        )

    dtrain, dval = encode_frame(train), encode_frame(val)

    def metrics(ev):
        logits, y = ev
        p = torch.softmax(torch.tensor(logits).float(), -1)[:, 1].numpy()
        m = {"fpr50": float((p[y == 0] >= 0.5).mean())}
        if 0 < y.sum() < len(y):
            m |= {
                "auc": roc_auc_score(y, p),
                "recall50": float((p[y == 1] >= 0.5).mean()),
            }
        return m

    args = TrainingArguments(
        output_dir=a.out,
        num_train_epochs=a.epochs,
        learning_rate=a.lr,
        max_steps=a.max_steps,
        gradient_checkpointing=a.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        per_device_train_batch_size=a.batch,
        gradient_accumulation_steps=a.accum,
        per_device_eval_batch_size=1,
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        warmup_steps=0 if a.max_steps == 1 else 0.06,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        bf16=torch.cuda.is_available(),
        eval_strategy="steps" if a.eval_steps else "no",
        eval_steps=a.eval_steps or 500,
        save_strategy="steps" if a.save_steps else "no",
        save_steps=a.save_steps or 500,
        save_only_model=True,
        remove_unused_columns=False,
        logging_steps=50,
        report_to=[],
        dataloader_num_workers=4,
        seed=a.seed,
    )
    trainer = DocumentTrainer(
        model=model,
        args=args,
        train_dataset=dtrain,
        eval_dataset=dval,
        data_collator=DocumentCollator(tok),
        compute_metrics=metrics,
        anchor_weight=a.anchor_weight,
        window_batch=a.window_batch,
        distill=a.distill,
    )
    trainer.train()

    # Score the complete scanner's inputs in fp32, just as deployed inference.
    from scripts.eval_ci import scan_scores

    model = trainer.accelerator.unwrap_model(trainer.model, keep_fp32_wrapper=False)
    model.float().eval()
    clf = Classifier(
        model,
        tok,
        0.5,
        a.max_length,
        device=model.device,
        context_tokens=a.context_tokens,
    )
    scored = scan_scores(
        val.assign(path=val.get("path", "validation")), clf, batch=a.window_batch
    )
    if a.scores_out:
        a.scores_out.parent.mkdir(parents=True, exist_ok=True)
        scored.to_parquet(a.scores_out, index=False)
    p = scored.score.to_numpy()
    y = val["label"].to_numpy()
    hard = (y == 0) & val["source"].isin(HARD).to_numpy()
    clf.threshold = thr = (
        a.threshold
        if a.threshold is not None
        else best_threshold(p[hard] if hard.any() else p[y == 0], a.fpr_target)
    )
    hits = p >= thr
    final = {
        "auc": float(roc_auc_score(y, p)),
        "threshold": thr,
        "hard_negatives": int(hard.sum()),
        "recall": float(hits[y == 1].mean()),
        "fpr": float(hits[y == 0].mean()),
        "hard_fpr": float(hits[hard].mean()) if hard.any() else None,
    }
    print(json.dumps(final, indent=1))
    clf.save(
        a.out,
        base=a.base,
        train_rows=len(train),
        epochs=float(trainer.state.epoch or 0),
        requested_epochs=a.epochs,
        optimizer_steps=trainer.state.global_step,
        lr=a.lr,
        val=final,
        training_windows="scanner_segments_document_max",
        negative_windows="average_of_max_and_mean",
        positive_windows="known_attack_spans_or_document_max",
        fpr_target=a.fpr_target,
        seed=a.seed,
        replay_anchor_weight=a.anchor_weight,
        training_window_batch=a.window_batch,
        training_context_tokens=a.context_tokens,
        train_last_layers=a.train_last_layers,
        trainable_parameters=sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        replay_anchor_margin_limit=12 if a.distill else 6,
        replay_anchor_scope="known_payload_or_document_margin",
        replay_anchor_objective=(
            "correct_teacher_kl_t2" if a.distill else "bounded_correct_margin_floor"
        ),
        replay_distill_minimum_correct_margin=4.5 if a.distill else None,
    )
    print(f"saved {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
