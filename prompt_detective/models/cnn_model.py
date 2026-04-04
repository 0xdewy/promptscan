#!/usr/bin/env python3
"""
CNN model for prompt injection detection.
"""

from typing import Any, Dict

import torch
import torch.nn as nn

from ..processors.word_processor import WordProcessor
from .base_model import BaseModel, BaseProcessor


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
            "model_type": "cnn",
        }

    @classmethod
    def load(cls, checkpoint_path: str, device: str = "cpu") -> "SimpleCNN":
        """Load model from checkpoint."""
        import pickle

        # Try to load with pickle first for compatibility
        try:
            with open(checkpoint_path, "rb") as f:
                checkpoint = pickle.load(f)
        except (pickle.UnpicklingError, RuntimeError):
            # If pickle fails, try torch.load with weights_only=False
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
            embedding_dim=checkpoint.get("embedding_dim", 64),
            num_filters=checkpoint.get("num_filters", 50),
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
        # Ensure model is on CPU and in eval mode for saving
        original_device = next(self.parameters()).device
        self.cpu()
        self.eval()

        # Save state dict separately to avoid pickle issues
        state_dict = self.state_dict()

        # Convert all tensors in state dict to CPU and make them contiguous
        for key in state_dict:
            state_dict[key] = state_dict[key].cpu().contiguous()

        checkpoint = {
            "model_type": "cnn",
            "model_state_dict": state_dict,
            "vocab_size": self.embedding.num_embeddings,
            "embedding_dim": self.embedding.embedding_dim,
            "num_filters": self.conv3.out_channels,
            "num_classes": self.fc[-1].out_features,
            "processor_config": processor.get_config() if processor else {},
            "pytorch_version": str(
                torch.__version__
            ),  # Convert to string for compatibility
            **metadata,
        }

        # Use pickle directly with protocol 2 for maximum compatibility
        import pickle

        with open(checkpoint_path, "wb") as f:
            pickle.dump(checkpoint, f, protocol=2)

        # Restore original device
        self.to(original_device)
