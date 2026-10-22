#!/usr/bin/env python3
"""
Transformer model (DistilBERT) for prompt injection detection.
"""

from typing import Any, Dict

import torch

from .base_model import BaseModel, BaseProcessor, finish_prediction

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

        # Suppress Hugging Face warnings during model loading
        import warnings
        import transformers

        # Save current logging level
        original_verbosity = transformers.logging.get_verbosity()

        try:
            # Set to error level to suppress warnings
            transformers.logging.set_verbosity_error()

            # Try to load model locally first, then from Hugging Face if needed
            try:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=num_classes,
                    local_files_only=True,
                    ignore_mismatched_sizes=True,  # Suppress size mismatch warnings
                )
            except Exception:
                # If local model not found, try to download it
                # (without local_files_only to allow download)
                try:
                    self.model = AutoModelForSequenceClassification.from_pretrained(
                        model_name,
                        num_labels=num_classes,
                        ignore_mismatched_sizes=True,  # Suppress size mismatch warnings
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load model '{model_name}'. Please check your internet connection.\n"
                        f"Original error: {e}"
                    )
        finally:
            # Restore original logging level
            transformers.logging.set_verbosity(original_verbosity)

        self.model_name = model_name
        self.num_classes = num_classes

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(**inputs).logits

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

            # Ensure all tensors have batch dimension
            for key in list(inputs.keys()):
                if inputs[key].dim() == 1:
                    inputs[key] = inputs[key].unsqueeze(0)

            outputs = self(inputs)

        return finish_prediction(outputs, "transformer", confidence_threshold, calibration_temperature)

    @classmethod
    def load(cls, checkpoint_path: str, device: str = "cpu") -> "TransformerModel":
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

        # Create model with warning suppression
        import warnings
        import transformers

        # Save current logging level
        original_verbosity = transformers.logging.get_verbosity()

        try:
            # Set to error level to suppress warnings
            transformers.logging.set_verbosity_error()

            model = cls(
                model_name=config.get("model_name", "distilbert-base-uncased"),
                num_classes=config.get("num_classes", 2),
            )

            # Load weights into the actual model (self.model), not the wrapper
            # Use strict=False to ignore mismatched keys (vocab layers vs classifier)
            model.model.load_state_dict(state_dict, strict=False)
            model.to(device)
            model.eval()
        finally:
            # Restore original logging level
            transformers.logging.set_verbosity(original_verbosity)

        # Create processor
        if "processor_config" in config:
            from ..processors.subword_processor import SubwordProcessor

            processor = SubwordProcessor.from_config(config["processor_config"])
        else:
            from ..processors.subword_processor import SubwordProcessor

            processor = SubwordProcessor(model_name=model.model_name)

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
        state_dict = self.model.state_dict()
        safetensors_path = base_path.with_suffix(".safetensors")
        save_file(state_dict, str(safetensors_path))

        # Save config as JSON
        config = {
            "model_type": "transformer",
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "processor_config": processor.get_config() if processor else {},
            "pytorch_version": str(torch.__version__),
            **metadata,
        }

        config_path = base_path.with_suffix(".config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
