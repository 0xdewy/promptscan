#!/usr/bin/env python3
"""
Word-level text processor for CNN and LSTM models.
"""

import re
from collections import Counter
from typing import Any, Dict, List

import torch

from ..models.base_model import BaseProcessor


class WordProcessor(BaseProcessor):
    """Word-level text processor for CNN and LSTM models."""

    def __init__(self, max_length=100, min_freq=2):
        self.max_length = max_length
        self.min_freq = min_freq
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        self.inverse_vocab = {0: "<PAD>", 1: "<UNK>"}
        self.next_id = 2

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return text.split()

    def build_vocab(self, texts: List[str]):
        """Build vocabulary from texts."""
        word_counts = Counter()
        for text in texts:
            words = self._tokenize(text)
            word_counts.update(words)

        for word, count in word_counts.items():
            if count >= self.min_freq:
                self.vocab[word] = self.next_id
                self.inverse_vocab[self.next_id] = word
                self.next_id += 1

    def encode(self, text: str) -> Dict[str, torch.Tensor]:
        """Convert text to token IDs."""
        words = self._tokenize(text)
        token_ids = []

        for word in words:
            token_ids.append(self.vocab.get(word, 1))  # 1 = <UNK>

        # Pad or truncate
        if len(token_ids) > self.max_length:
            token_ids = token_ids[: self.max_length]
        else:
            token_ids = token_ids + [0] * (self.max_length - len(token_ids))

        return {"input_ids": torch.tensor(token_ids, dtype=torch.long)}

    def batch_encode(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Encode multiple texts."""
        batch_ids = []
        for text in texts:
            words = self._tokenize(text)
            token_ids = []

            for word in words:
                token_ids.append(self.vocab.get(word, 1))

            if len(token_ids) > self.max_length:
                token_ids = token_ids[: self.max_length]
            else:
                token_ids = token_ids + [0] * (self.max_length - len(token_ids))

            batch_ids.append(token_ids)

        return {"input_ids": torch.tensor(batch_ids, dtype=torch.long)}

    def get_config(self) -> Dict[str, Any]:
        """Get processor configuration."""
        return {
            "max_length": self.max_length,
            "min_freq": self.min_freq,
            "vocab": self.vocab,
            "vocab_size": len(self.vocab),
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "WordProcessor":
        """Create processor from configuration."""
        processor = cls(
            max_length=config.get("max_length", 100), min_freq=config.get("min_freq", 2)
        )
        processor.vocab = config["vocab"]
        processor.inverse_vocab = {v: k for k, v in config["vocab"].items()}
        processor.next_id = len(config["vocab"])
        return processor
