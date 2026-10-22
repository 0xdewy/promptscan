#!/usr/bin/env python3
"""
CNN model for prompt injection detection.
"""

from typing import Any, Dict

import torch
import torch.nn as nn

from ..processors.word_processor import WordProcessor
from .base_model import BaseModel, BaseProcessor, finish_prediction


class SimpleCNN(BaseModel):
    """Simple CNN model for prompt injection detection."""

    def __init__(self, vocab_size, embedding_dim=64, num_filters=50, num_classes=2):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # CNN with multiple filter sizes (3, 4, 5)
        self.conv3 = nn.Conv1d(embedding_dim, num_filters, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(embedding_dim, num_filters, kernel_size=4, padding=2)
        self.conv5 = nn.Conv1d(embedding_dim, num_filters, kernel_size=5, padding=2)

        self.dropout = nn.Dropout(0.3)

        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(num_filters * 3, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = inputs["input_ids"]
        # x shape: (batch_size, seq_len)
        embeddings = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        embeddings = embeddings.transpose(1, 2)  # (batch_size, embedding_dim, seq_len)

        # Apply convolutions
        conv3_out = torch.relu(self.conv3(embeddings))
        conv4_out = torch.relu(self.conv4(embeddings))
        conv5_out = torch.relu(self.conv5(embeddings))

        # Max pooling
        pooled3 = torch.max(conv3_out, dim=2)[0]
        pooled4 = torch.max(conv4_out, dim=2)[0]
        pooled5 = torch.max(conv5_out, dim=2)[0]

        # Concatenate
        concatenated = torch.cat([pooled3, pooled4, pooled5], dim=1)
        concatenated = self.dropout(concatenated)

        # Classify
        logits = self.fc(concatenated)
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

        return finish_prediction(outputs, "cnn", confidence_threshold, calibration_temperature)

    @classmethod
    def load(cls, checkpoint_path: str, device: str = "cpu") -> "SimpleCNN":
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
            embedding_dim=config.get("embedding_dim", 64),
            num_filters=config.get("num_filters", 50),
            num_classes=config.get("num_classes", 2),
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

        # Ensure model is on CPU and in eval mode for saving
        original_device = next(self.parameters()).device
        self.cpu()
        self.eval()

        # Get base path without extension
        checkpoint_path = Path(checkpoint_path)
        base_path = checkpoint_path.parent / checkpoint_path.stem

        # Save model weights as safetensors
        state_dict = self.state_dict()
        safetensors_path = base_path.with_suffix(".safetensors")
        save_file(state_dict, str(safetensors_path))

        # Save config as JSON
        config = {
            "model_type": "cnn",
            "vocab_size": self.embedding.num_embeddings,
            "embedding_dim": self.embedding.embedding_dim,
            "num_filters": self.conv3.out_channels,
            "num_classes": self.fc[-1].out_features,
            "processor_config": processor.get_config() if processor else {},
            "pytorch_version": str(torch.__version__),
            **metadata,
        }

        config_path = base_path.with_suffix(".config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        # Restore original device
        self.to(original_device)
