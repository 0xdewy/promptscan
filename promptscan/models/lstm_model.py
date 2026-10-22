#!/usr/bin/env python3
"""
LSTM model for prompt injection detection.
"""

from typing import Any, Dict

import torch
import torch.nn as nn

from ..processors.word_processor import WordProcessor
from .base_model import BaseModel, BaseProcessor, finish_prediction


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
        # Read the state at the last real token, not at the trailing padding.
        last_index = (input_ids != 0).sum(dim=1).clamp(min=1) - 1
        last_hidden = lstm_out[torch.arange(input_ids.size(0), device=input_ids.device), last_index]
        last_hidden = self.dropout(last_hidden)

        logits = self.classifier(last_hidden)
        return logits

    def predict(
        self,
        text: str,
        processor: BaseProcessor,
        confidence_threshold: float = 0.5,
        calibration_temperature: float = 1.0,
    ) -> Dict[str, Any]:
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

        return finish_prediction(outputs, "lstm", confidence_threshold, calibration_temperature)

    @classmethod
    def load(cls, checkpoint_path: str, device: str = "cpu") -> "LSTMModel":
        """Load model from checkpoint."""
        import json
        from pathlib import Path
        from safetensors.torch import load_file

        checkpoint_dir = Path(checkpoint_path).parent
        checkpoint_name = Path(checkpoint_path).stem

        # Load config
        config_path = checkpoint_dir / f"{checkpoint_name}.config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            config = json.load(f)

        # Load model weights from safetensors
        safetensors_path = checkpoint_dir / f"{checkpoint_name}.safetensors"
        if not safetensors_path.exists():
            raise FileNotFoundError(f"Model weights not found: {safetensors_path}")

        state_dict = load_file(str(safetensors_path), device=device)

        # Create model
        model = cls(
            vocab_size=config["vocab_size"],
            embedding_dim=config.get("embedding_dim", 128),
            hidden_dim=config.get("hidden_dim", 128),
            num_layers=config.get("num_layers", 2),
            num_classes=config.get("num_classes", 2),
            dropout=config.get("dropout", 0.3),
        )

        # Load weights
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        # Create processor from config
        if "processor_config" in config:
            processor = WordProcessor.from_config(config["processor_config"])
        else:
            # Fallback for older config format
            processor_config = {
                "max_length": config.get("max_length", 100),
                "min_freq": config.get("min_freq", 2),
                "vocab": config.get("vocab", {}),
            }
            processor = WordProcessor.from_config(processor_config)

        return model, processor

    def save(self, checkpoint_path: str, processor=None, **metadata):
        """Save model to checkpoint."""
        import json
        from pathlib import Path
        from safetensors.torch import save_file

        # Get base path without extension
        checkpoint_path = Path(checkpoint_path)
        base_path = checkpoint_path.parent / checkpoint_path.stem

        # Save model weights as safetensors
        state_dict = self.state_dict()
        safetensors_path = base_path.with_suffix(".safetensors")
        save_file(state_dict, str(safetensors_path))

        # Save config as JSON
        config = {
            "model_type": "lstm",
            "vocab_size": self.embedding.num_embeddings,
            "embedding_dim": self.embedding.embedding_dim,
            "hidden_dim": self.lstm.hidden_size,
            "num_layers": self.lstm.num_layers,
            "num_classes": self.classifier[-1].out_features,
            "dropout": self.dropout.p,
            "processor_config": processor.get_config() if processor else {},
            "pytorch_version": str(torch.__version__),
            **metadata,
        }

        config_path = base_path.with_suffix(".config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
