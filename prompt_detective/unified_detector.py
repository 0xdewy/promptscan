#!/usr/bin/env python3
"""
Unified detector interface for all model types.
"""

import os
from typing import Any, Dict, List

from . import get_model_path
from .ensemble.detector import EnsembleDetector
from .models.cnn_model import SimpleCNN
from .models.lstm_model import LSTMModel
from .models.transformer_model import TransformerModel


class UnifiedDetector:
    """Unified interface for all detector types."""

    def __init__(self, model_type="cnn", model_path=None, device="cpu", **kwargs):
        """
        Initialize detector.

        Args:
            model_type: "cnn", "lstm", "transformer", or "ensemble"
            model_path: Path to model checkpoint (optional for ensemble)
            device: "cpu", "cuda", or "auto"
            **kwargs: Additional arguments for ensemble
        """
        from .utils.device import get_device

        self.model_type = model_type
        self.device = get_device(device)  # Convert "auto" to "cpu" or "cuda"

        if model_type == "ensemble":
            model_dir = kwargs.get("model_dir", model_path)
            if model_dir is None:
                # Use package models directory
                package_dir = os.path.dirname(os.path.dirname(__file__))
                model_dir = os.path.join(package_dir, "models", "checkpoints")
            self.detector = EnsembleDetector.from_pretrained(
                model_dir=model_dir,
                voting_strategy=kwargs.get("voting_strategy", "majority"),
                device=self.device,
            )
        else:
            if model_path is None:
                # Default model paths
                if model_type == "cnn":
                    model_path = str(get_model_path("cnn_best.pt"))
                elif model_type == "lstm":
                    model_path = str(get_model_path("lstm_best.pt"))
                elif model_type == "transformer":
                    model_path = str(get_model_path("transformer_best.pt"))

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")

            # Load appropriate model
            if model_type == "cnn":
                self.model, self.processor = SimpleCNN.load(model_path, self.device)
            elif model_type == "lstm":
                self.model, self.processor = LSTMModel.load(model_path, self.device)
            elif model_type == "transformer":
                self.model, self.processor = TransformerModel.load(
                    model_path, self.device
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")

    def predict(self, text: str) -> Dict[str, Any]:
        """Predict if text contains prompt injection."""
        if self.model_type == "ensemble":
            return self.detector.predict(text)
        else:
            return self.model.predict(text, self.processor)

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Predict for multiple texts."""
        if self.model_type == "ensemble":
            return self.detector.predict_batch(texts)
        else:
            results = []
            for text in texts:
                results.append(self.predict(text))
            return results

    def get_info(self) -> Dict[str, Any]:
        """Get detector information."""
        if self.model_type == "ensemble":
            return {
                "type": "ensemble",
                "models": self.detector.get_model_info(),
                "voting_strategy": self.detector.voting_strategy,
            }
        else:
            return {
                "type": self.model_type,
                "parameters": sum(p.numel() for p in self.model.parameters()),
                "device": str(self.model.get_device()),
            }
