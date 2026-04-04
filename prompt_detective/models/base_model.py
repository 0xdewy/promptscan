#!/usr/bin/env python3
"""
Base model interface for prompt injection detection.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import torch
import torch.nn as nn


class BaseModel(ABC, nn.Module):
    """Abstract base class for all prompt injection detection models."""

    @abstractmethod
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through the model."""
        pass

    @abstractmethod
    def predict(self, text: str, processor) -> Dict[str, Any]:
        """Predict if text contains prompt injection."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, checkpoint_path: str, device: str = "cpu") -> "BaseModel":
        """Load model from checkpoint."""
        pass

    @abstractmethod
    def save(self, checkpoint_path: str, processor=None, **metadata):
        """Save model to checkpoint."""
        pass

    def get_device(self) -> str:
        """Get the device the model is on."""
        return next(self.parameters()).device


class BaseProcessor(ABC):
    """Abstract base class for text processors."""

    @abstractmethod
    def encode(self, text: str) -> Dict[str, torch.Tensor]:
        """Encode text into model inputs."""
        pass

    @abstractmethod
    def batch_encode(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Encode multiple texts."""
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get processor configuration."""
        pass

    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict[str, Any]) -> "BaseProcessor":
        """Create processor from configuration."""
        pass
