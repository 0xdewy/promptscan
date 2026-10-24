"""The promptscan classifier: a fine-tuned DeBERTa-v3 that outputs P(injection).

Long inputs are scored in overlapping windows and the max window score wins, so an
injection buried in page 3 of a document is not diluted by pages 1 and 2.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_REPO = "0xdewy/promptscan"


class Classifier:
    def __init__(self, model, tokenizer, threshold: float, max_length: int, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.threshold = threshold
        self.max_length = max_length
        self.device = device
        labels = {v.lower(): k for k, v in model.config.id2label.items()}
        self.pos = labels.get("injection", 1)

    @classmethod
    def load(cls, path: str | None = None, device: str = "cpu") -> "Classifier":
        path = path or DEFAULT_REPO
        model = AutoModelForSequenceClassification.from_pretrained(path, dtype=torch.float32)
        tok = AutoTokenizer.from_pretrained(path)
        meta = getattr(model.config, "promptscan", None) or {}
        return cls(model, tok, float(meta.get("threshold", 0.5)), int(meta.get("max_length", 512)), device)

    @torch.no_grad()
    def score(self, texts: list[str], batch: int = 16) -> list[float]:
        """P(injection) per text; max over overlapping windows for long texts."""
        enc = self.tokenizer([t[:200_000] for t in texts], truncation=True, max_length=self.max_length,
                             stride=self.max_length // 4, return_overflowing_tokens=True)
        owner = enc["overflow_to_sample_mapping"]
        scores = [0.0] * len(texts)
        for s in range(0, len(owner), batch):
            chunk = self.tokenizer.pad(
                {k: enc[k][s:s + batch] for k in ("input_ids", "attention_mask")}, return_tensors="pt").to(self.device)
            p = torch.softmax(self.model(**chunk).logits.float(), -1)[:, self.pos].tolist()
            for j, pj in enumerate(p):
                o = owner[s + j]
                scores[o] = max(scores[o], pj)
        return scores

    def predict(self, text: str) -> dict[str, Any]:
        p = self.score([text])[0]
        hit = p >= self.threshold
        return {"prediction": "INJECTION" if hit else "SAFE", "confidence": p if hit else 1 - p,
                "score": p, "threshold": self.threshold}

    def save(self, path: str, **meta) -> None:
        self.model.config.promptscan = {"threshold": self.threshold, "max_length": self.max_length, **meta}
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        Path(path, "promptscan.json").write_text(json.dumps(self.model.config.promptscan, indent=1))
