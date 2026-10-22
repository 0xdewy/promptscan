#!/usr/bin/env python3
"""
Base model interface for prompt injection detection.
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


def finish_prediction(
    logits: torch.Tensor,
    model_type: str,
    confidence_threshold: float = 0.5,
    calibration_temperature: float = 1.0,
) -> Dict[str, Any]:
    """Turn a (1, 2) logit tensor into the prediction dict every model returns."""
    probabilities = torch.softmax(logits / max(calibration_temperature, 1e-6), dim=1)[0]
    p_injection = probabilities[1].item()
    is_injection = p_injection >= confidence_threshold
    return {
        "prediction": "INJECTION" if is_injection else "SAFE",
        "confidence": p_injection if is_injection else probabilities[0].item(),
        "class": 1 if is_injection else 0,
        "probabilities": probabilities.cpu().tolist(),
        "class_probabilities": {"SAFE": probabilities[0].item(), "INJECTION": p_injection},
        "model_type": model_type,
    }


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

    @classmethod
    def load_from_hf(
        cls,
        repo_id: str,
        model_dir: str = "",
        filename: str = "model.safetensors",
        config_filename: str = "config.json",
        device: str = "cpu",
        token: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> Tuple["BaseModel", "BaseProcessor"]:
        """
        Load model from Hugging Face Hub.

        Args:
            repo_id: Hugging Face repository ID (e.g., "0xdewy/promptscan")
            model_dir: Directory within the repository (e.g., "cnn", "lstm")
            filename: Name of the model weights file
            config_filename: Name of the config file
            device: Device to load the model on
            token: Hugging Face token (for private repos)
            cache_dir: Cache directory for downloaded files

        Returns:
            Tuple of (model, processor)
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "huggingface-hub is required to load models from Hugging Face Hub. "
                "Install with: pip install huggingface-hub"
            )

        # Download model files
        if cache_dir is None:
            cache_dir = os.environ.get(
                "HF_HOME", os.path.expanduser("~/.cache/huggingface")
            )

        # Construct full paths in repo
        if model_dir:
            model_path_in_repo = f"{model_dir}/{filename}"
            config_path_in_repo = f"{model_dir}/{config_filename}"
        else:
            model_path_in_repo = filename
            config_path_in_repo = config_filename

        try:
            # Download safetensors file
            safetensors_path = hf_hub_download(
                repo_id=repo_id,
                filename=model_path_in_repo,
                cache_dir=cache_dir,
                token=token,
                force_filename=f"{repo_id.replace('/', '_')}_{model_dir or 'model'}.safetensors",
            )

            # Download config file
            config_path = hf_hub_download(
                repo_id=repo_id,
                filename=config_path_in_repo,
                cache_dir=cache_dir,
                token=token,
                force_filename=f"{repo_id.replace('/', '_')}_{model_dir or 'model'}.config.json",
            )

            # Load using the standard load method
            return cls.load(str(Path(safetensors_path).with_suffix("")), device)

        except Exception as e:
            raise RuntimeError(
                f"Failed to load model from Hugging Face Hub (repo_id={repo_id}, "
                f"model_dir={model_dir}): {e}"
            )


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
