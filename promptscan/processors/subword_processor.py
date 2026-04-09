#!/usr/bin/env python3
"""
Subword text processor for transformer models.
"""

from typing import Any, Dict, List

import torch

from ..models.base_model import BaseProcessor

# Try to import transformers, but don't fail if not available
try:
    from transformers import AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None


class SubwordProcessor(BaseProcessor):
    """Subword text processor for transformer models."""

    def __init__(self, model_name="distilbert-base-uncased", max_length=128):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Transformers library is not installed. "
                "Install with: pip install transformers"
            )
        # Use local models only to avoid HF Hub warnings
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, local_files_only=True
            )
        except Exception as e:
            raise RuntimeError(
                f"Tokenizer for '{model_name}' not found locally. Please download it first.\n"
                f'Run: python -c "from transformers import AutoTokenizer; '
                f"AutoTokenizer.from_pretrained('{model_name}')\"\n"
                f"Original error: {e}"
            )
        self.max_length = max_length

    def encode(self, text: str) -> Dict[str, torch.Tensor]:
        """Convert text to transformer inputs."""
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v for k, v in encoding.items()}

    def batch_encode(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Encode multiple texts."""
        encoding = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v for k, v in encoding.items()}

    def get_config(self) -> Dict[str, Any]:
        """Get processor configuration."""
        return {
            "model_name": self.tokenizer.name_or_path,
            "max_length": self.max_length,
            "vocab_size": self.tokenizer.vocab_size,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SubwordProcessor":
        """Create processor from configuration."""
        return cls(
            model_name=config.get("model_name", "distilbert-base-uncased"),
            max_length=config.get("max_length", 128),
        )
