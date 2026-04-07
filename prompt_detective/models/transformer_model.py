#!/usr/bin/env python3
"""
Transformer model (DistilBERT) for prompt injection detection.
"""

from typing import Any, Dict

import torch

from .base_model import BaseModel, BaseProcessor

# Try to import transformers, but don't fail if not available
try:
    from transformers import AutoModelForSequenceClassification

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoModelForSequenceClassification = None


class TransformerModel(BaseModel):
    """Transformer model for prompt injection detection."""

    def __init__(self, model_name="distilbert-base-uncased", num_classes=2):
        super().__init__()
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Transformers library is required for TransformerModel. "
                "Install with: pip install transformers"
            )

        # Use local models only to avoid HF Hub warnings
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=num_classes, local_files_only=True
            )
        except Exception as e:
            raise RuntimeError(
                f"Model '{model_name}' not found locally. Please download it first.\n"
                f'Run: python -c "from transformers import AutoModelForSequenceClassification; '
                f"AutoModelForSequenceClassification.from_pretrained('{model_name}')\"\n"
                f"Original error: {e}"
            )
        self.model_name = model_name
        self.num_classes = num_classes

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(**inputs).logits

    def predict(self, text: str, processor: BaseProcessor) -> Dict[str, Any]:
        """Predict if text contains prompt injection."""
        self.eval()
        with torch.no_grad():
            inputs = processor.encode(text)
            device = self.get_device()
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Ensure all tensors have batch dimension
            for key in list(inputs.keys()):
                if inputs[key].dim() == 1:
                    inputs[key] = inputs[key].unsqueeze(0)

            outputs = self(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            pred_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][int(pred_class)].item()

        return {
            "prediction": "INJECTION" if pred_class == 1 else "SAFE",
            "confidence": confidence,
            "class": pred_class,
            "probabilities": probabilities[0].cpu().numpy().tolist(),
            "model_type": "transformer",
        }

    @classmethod
    def load(cls, checkpoint_path: str, device: str = "cpu") -> "TransformerModel":
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
            model_name=checkpoint.get("model_name", "distilbert-base-uncased"),
            num_classes=checkpoint.get("num_classes", 2),
        )

        # Load weights into the actual model (self.model), not the wrapper
        model.model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        # Create processor
        if "processor_config" in checkpoint:
            from ..processors.subword_processor import SubwordProcessor

            processor = SubwordProcessor.from_config(checkpoint["processor_config"])
        else:
            from ..processors.subword_processor import SubwordProcessor

            processor = SubwordProcessor(model_name=model.model_name)

        return model, processor

    def save(self, checkpoint_path: str, processor=None, **metadata):
        """Save model to checkpoint."""
        checkpoint = {
            "model_type": "transformer",
            "model_state_dict": self.model.state_dict(),
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "processor_config": processor.get_config() if processor else {},
            **metadata,
        }

        torch.save(checkpoint, checkpoint_path, _use_new_zipfile_serialization=False)
