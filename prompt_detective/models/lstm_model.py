#!/usr/bin/env python3
"""
LSTM model for prompt injection detection.
"""

from typing import Any, Dict

import torch
import torch.nn as nn

from ..processors.word_processor import WordProcessor
from .base_model import BaseModel, BaseProcessor


class LSTMModel(BaseModel):
    """LSTM model for prompt injection detection."""

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=128,
        num_layers=2,
        num_classes=2,
        dropout=0.3,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = inputs["input_ids"]
        embeddings = self.embedding(input_ids)

        lstm_out, _ = self.lstm(embeddings)
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]
        last_hidden = self.dropout(last_hidden)

        logits = self.classifier(last_hidden)
        return logits

    def predict(self, text: str, processor: BaseProcessor) -> Dict[str, Any]:
        """Predict if text contains prompt injection."""
        self.eval()
        with torch.no_grad():
            inputs = processor.encode(text)
            device = self.get_device()
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Add batch dimension if needed
            if inputs["input_ids"].dim() == 1:
                inputs["input_ids"] = inputs["input_ids"].unsqueeze(0)

            outputs = self(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            pred_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][int(pred_class)].item()

        return {
            "prediction": "INJECTION" if pred_class == 1 else "SAFE",
            "confidence": confidence,
            "class": pred_class,
            "probabilities": probabilities[0].cpu().numpy().tolist(),
            "model_type": "lstm",
        }

    @classmethod
    def load(cls, checkpoint_path: str, device: str = "cpu") -> "LSTMModel":
        """Load model from checkpoint."""
        import pickle

        # Try to load with weights_only=True first (safer)
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=True
            )
        except (pickle.UnpicklingError, RuntimeError):
            # If that fails, try with weights_only=False
            # (for old models or compatibility issues)
            import warnings

            warnings.warn(
                f"Loading model with weights_only=False - ensure {checkpoint_path} "
                "is from a trusted source",
                stacklevel=2,
            )
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )

        # Create model
        model = cls(
            vocab_size=checkpoint["vocab_size"],
            embedding_dim=checkpoint.get("embedding_dim", 128),
            hidden_dim=checkpoint.get("hidden_dim", 128),
            num_layers=checkpoint.get("num_layers", 2),
            num_classes=checkpoint.get("num_classes", 2),
        )

        # Load weights
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        # Create processor - handle both old and new checkpoint formats
        if "processor_config" in checkpoint:
            processor = WordProcessor.from_config(checkpoint["processor_config"])
        else:
            # Backward compatibility: create config from old checkpoint format
            processor_config = {
                "max_length": checkpoint.get("max_length", 100),
                "min_freq": checkpoint.get("min_freq", 2),
                "vocab": checkpoint["vocab"],
            }
            processor = WordProcessor.from_config(processor_config)

        return model, processor

    def save(self, checkpoint_path: str, processor=None, **metadata):
        """Save model to checkpoint."""
        checkpoint = {
            "model_type": "lstm",
            "model_state_dict": self.state_dict(),
            "vocab_size": self.embedding.num_embeddings,
            "embedding_dim": self.embedding.embedding_dim,
            "hidden_dim": self.lstm.hidden_size,
            "num_layers": self.lstm.num_layers,
            "num_classes": self.classifier[-1].out_features,
            "dropout": self.dropout.p,
            "processor_config": processor.get_config() if processor else {},
            **metadata,
        }

        torch.save(checkpoint, checkpoint_path, _use_new_zipfile_serialization=False)
