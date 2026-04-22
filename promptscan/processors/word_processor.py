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

    def __init__(self, max_length=256, min_freq=5, max_vocab_size=30000):
        self.max_length = max_length
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        self.inverse_vocab = {0: "<PAD>", 1: "<UNK>"}
        self.next_id = 2

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return text.split()

    def build_vocab(self, texts: List[str]):
        """Build vocabulary from texts, respecting max_vocab_size limit."""
        word_counts = Counter()
        for text in texts:
            words = self._tokenize(text)
            word_counts.update(words)

        filtered = [(w, c) for w, c in word_counts.items() if c >= self.min_freq]
        filtered.sort(key=lambda x: -x[1])

        slots_available = max(0, self.max_vocab_size - len(self.vocab))
        if slots_available <= 0:
            return

        if len(filtered) > slots_available:
            filtered = filtered[:slots_available]

        for word, count in filtered:
            self.vocab[word] = self.next_id
            self.inverse_vocab[self.next_id] = word
            self.next_id += 1

    def encode(self, text: str) -> Dict[str, torch.Tensor]:
        """Convert text to token IDs."""
        words = self._tokenize(text)
        token_ids = []

        for word in words:
            token_ids.append(self.vocab.get(word, 1))

        actual_length = min(len(token_ids), self.max_length)

        if len(token_ids) > self.max_length:
            token_ids = token_ids[: self.max_length]
        else:
            token_ids = token_ids + [0] * (self.max_length - len(token_ids))

        attention_mask = [1] * actual_length + [0] * (self.max_length - actual_length)

        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def batch_encode(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Encode multiple texts."""
        batch_ids = []
        batch_masks = []
        for text in texts:
            words = self._tokenize(text)
            token_ids = []

            for word in words:
                token_ids.append(self.vocab.get(word, 1))

            actual_length = min(len(token_ids), self.max_length)

            if len(token_ids) > self.max_length:
                token_ids = token_ids[: self.max_length]
            else:
                token_ids = token_ids + [0] * (self.max_length - len(token_ids))

            attention_mask = [1] * actual_length + [0] * (
                self.max_length - actual_length
            )

            batch_ids.append(token_ids)
            batch_masks.append(attention_mask)

        return {
            "input_ids": torch.tensor(batch_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_masks, dtype=torch.long),
        }

    def get_config(self) -> Dict[str, Any]:
        """Get processor configuration."""
        return {
            "max_length": self.max_length,
            "min_freq": self.min_freq,
            "max_vocab_size": self.max_vocab_size,
            "vocab": self.vocab,
            "vocab_size": len(self.vocab),
            "next_id": self.next_id,
            "embedding_size": self.next_id,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "WordProcessor":
        """Create processor from configuration."""
        processor = cls(
            max_length=config.get("max_length", 256),
            min_freq=config.get("min_freq", 5),
            max_vocab_size=config.get("max_vocab_size", 30000),
        )
        processor.vocab = config["vocab"]
        processor.inverse_vocab = {v: k for k, v in config["vocab"].items()}

        if "next_id" in config:
            processor.next_id = config["next_id"]
        else:
            max_token_id = max(v for v in processor.vocab.values() if isinstance(v, int))
            processor.next_id = max_token_id + 1

        if "embedding_size" in config:
            required_size = config["embedding_size"]
        else:
            required_size = processor.next_id

        if required_size > processor.max_vocab_size:
            processor.max_vocab_size = required_size

        return processor

    def validate_training_data(self, texts: List[str]) -> Dict[str, Any]:
        """Validate that training data only uses tokens within model's embedding capacity."""
        max_token_id = max(v for v in self.vocab.values() if isinstance(v, int))
        embedding_size = self.next_id

        out_of_range_count = 0
        max_id_found = 0
        problematic_texts = []

        for text in texts:
            words = self._tokenize(text)
            for word in words:
                if word in self.vocab:
                    token_id = self.vocab[word]
                else:
                    token_id = 1
                if token_id > max_id_found:
                    max_id_found = token_id

        for text in texts:
            encoded = self.encode(text)
            ids = encoded["input_ids"].tolist()
            for id_val in ids:
                if id_val != 0 and id_val >= embedding_size:
                    out_of_range_count += 1
                    if len(problematic_texts) < 5:
                        problematic_texts.append(text[:50])
                    break

        return {
            "valid": out_of_range_count == 0,
            "max_id_found": max_id_found,
            "embedding_size": embedding_size,
            "out_of_range_count": out_of_range_count,
            "problematic_samples": len(problematic_texts),
            "example_problems": problematic_texts,
        }