"""
Text processing utilities for prompt injection detection.
"""

import html
import re
from collections import Counter
from typing import List


def clean_text(text: str) -> str:
    """
    Clean text by:
    1. Unescaping HTML entities
    2. Removing HTML tags
    3. Normalizing whitespace
    4. Stripping extra quotes
    """
    if not text:
        return ""

    # Unescape HTML entities
    text = html.unescape(text)

    # Remove HTML tags (simple regex approach)
    text = re.sub(r"<[^>]+>", "", text)

    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)

    # Normalize whitespace
    text = " ".join(text.split())

    # Strip surrounding quotes if they exist
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]

    return text.strip()


class SimpleTextProcessor:
    """Minimal text processor for prompt injection detection."""

    def __init__(self, max_length=100):
        self.max_length = max_length
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        self.inverse_vocab = {0: "<PAD>", 1: "<UNK>"}
        self.next_id = 2

    def build_vocab(self, texts: List[str], min_freq=2):
        """Build vocabulary from texts."""
        word_counts = Counter()
        for text in texts:
            words = self._tokenize(text)
            word_counts.update(words)

        for word, count in word_counts.items():
            if count >= min_freq:
                self.vocab[word] = self.next_id
                self.inverse_vocab[self.next_id] = word
                self.next_id += 1

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return text.split()

    def encode(self, text: str) -> List[int]:
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

        return token_ids
