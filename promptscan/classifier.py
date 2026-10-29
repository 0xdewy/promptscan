"""The promptscan classifier: a fine-tuned DeBERTa-v3 that outputs P(injection).

Long inputs are scored in overlapping windows and the max window score wins, so an
injection buried in page 3 of a document is not diluted by pages 1 and 2.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .windows import encode_windows

DEFAULT_REPO = "0xdewy/promptscan"
DEFAULT_REVISION = "a4ab3dc8d3b84f70ae4c416883b2ba524c39d323"


class Classifier:
    def __init__(
        self,
        model,
        tokenizer,
        threshold: float,
        max_length: int,
        device: str = "cpu",
        context_tokens: int = 0,
    ):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.threshold = threshold
        self.max_length = max_length
        self.device = device
        self.context_tokens = context_tokens
        labels = {v.lower(): k for k, v in model.config.id2label.items()}
        self.pos = labels.get("injection", 1)

    @classmethod
    def load(
        cls,
        path: str | None = None,
        device: str = "cpu",
        revision: str | None = None,
    ) -> "Classifier":
        path = path or DEFAULT_REPO
        local = Path(path).is_dir()
        revision = revision or (DEFAULT_REVISION if path == DEFAULT_REPO else None)
        kwargs = {} if local else {"revision": revision}
        model = AutoModelForSequenceClassification.from_pretrained(
            path,
            dtype=torch.float32,
            use_safetensors=True,
            trust_remote_code=False,
            **kwargs,
        )
        resolved = getattr(model.config, "_commit_hash", None) or revision
        tok = AutoTokenizer.from_pretrained(
            path,
            trust_remote_code=False,
            **({} if local else {"revision": resolved}),
        )
        meta = getattr(model.config, "promptscan", None) or {}
        classifier = cls(
            model,
            tok,
            float(meta.get("threshold", 0.5)),
            int(meta.get("max_length", 512)),
            device,
            context_tokens=int(meta.get("context_tokens", 0)),
        )
        classifier.identity = {
            "source": str(path),
            "revision": resolved if not local else None,
        }
        if local:
            digest = hashlib.sha256()
            names = {
                "config.json",
                "model.safetensors.index.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "added_tokens.json",
                "vocab.txt",
                "vocab.json",
                "merges.txt",
                "spm.model",
                "tokenizer.model",
                "sentencepiece.bpe.model",
            }
            names.update(file.name for file in Path(path).glob("*.safetensors"))
            hashed = []
            for name in sorted(names):
                file = Path(path, name)
                if file.exists():
                    hashed.append(name)
                    digest.update(name.encode())
                    with file.open("rb") as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(block)
            classifier.identity["artifact_sha256"] = digest.hexdigest()
            classifier.identity["hashed_files"] = hashed
        return classifier

    @torch.no_grad()
    def score(self, texts: list[str], batch: int = 16) -> list[float]:
        """P(injection) per text; max over overlapping windows for long texts."""
        if not texts:
            return []
        enc = encode_windows(
            self.tokenizer, texts, self.max_length, self.context_tokens
        )
        owner = enc["overflow_to_sample_mapping"]
        scores = [0.0] * len(texts)
        # Group similarly sized windows so short prompts do not get padded to
        # 512 tokens alongside long documents. Ownership still follows encoding.
        order = sorted(range(len(owner)), key=lambda i: len(enc["input_ids"][i]))
        for s in range(0, len(order), batch):
            indices = order[s : s + batch]
            chunk = self.tokenizer.pad(
                {
                    k: [enc[k][i] for i in indices]
                    for k in ("input_ids", "attention_mask")
                },
                return_tensors="pt",
            ).to(self.device)
            p = torch.softmax(self.model(**chunk).logits.float(), -1)[
                :, self.pos
            ].tolist()
            for i, pj in zip(indices, p, strict=True):
                o = owner[i]
                scores[o] = max(scores[o], pj)
        return scores

    def predict(self, text: str) -> dict[str, Any]:
        p = self.score([text])[0]
        hit = p >= self.threshold
        return {
            "prediction": "INJECTION" if hit else "SAFE",
            "confidence": p if hit else 1 - p,
            "score": p,
            "threshold": self.threshold,
        }

    def save(self, path: str, **meta) -> None:
        from .scan import SCANNER_POLICY_VERSION

        settings = {
            "threshold": self.threshold,
            "max_length": self.max_length,
            "context_tokens": self.context_tokens,
            "scanner_policy": SCANNER_POLICY_VERSION,
            **meta,
        }
        serialized = json.dumps(settings, indent=1, allow_nan=False)
        self.model.config.promptscan = settings
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        Path(path, "promptscan.json").write_text(serialized)
