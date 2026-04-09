#!/usr/bin/env python3
"""
Ensemble detector with parallel inference and consensus voting.
"""

import concurrent.futures
from pathlib import Path
from typing import Any, Dict, List

from ..models.cnn_model import SimpleCNN
from ..models.lstm_model import LSTMModel
from ..models.transformer_model import TransformerModel
from .voting import VotingStrategies


class EnsembleDetector:
    """Ensemble detector with multiple models and voting."""

    def __init__(
        self,
        model_configs: List[Dict[str, Any]],
        voting_strategy: str = "majority",
        device: str = "cpu",
        max_workers: int = 3,
    ):
        """
        Initialize ensemble detector.

        Args:
            model_configs: List of model configurations, each with:
                - type: "cnn", "lstm", or "transformer"
                - checkpoint_path: Path to model checkpoint
                - weight: Optional weight for weighted voting
            voting_strategy: "majority", "weighted", "confidence", or "soft"
            device: "cpu" or "cuda"
            max_workers: Maximum number of parallel workers
        """
        self.models = []
        self.processors = []
        self.weights = []
        self.model_types = []
        self.voting_strategy = voting_strategy
        self.device = device
        self.max_workers = max_workers

        # Load models
        for config in model_configs:
            model_type = config["type"]
            checkpoint_path = config["checkpoint_path"]
            weight = config.get("weight", 1.0)

            if model_type == "cnn":
                model, processor = SimpleCNN.load(checkpoint_path, device)
            elif model_type == "lstm":
                model, processor = LSTMModel.load(checkpoint_path, device)
            elif model_type == "transformer":
                model, processor = TransformerModel.load(checkpoint_path, device)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            self.models.append(model)
            self.processors.append(processor)
            self.weights.append(weight)
            self.model_types.append(model_type)

        print(f"Loaded {len(self.models)} models for ensemble detection")
        print(f"Models: {[config['type'] for config in model_configs]}")
        print(f"Voting strategy: {voting_strategy}")

    def _predict_single(self, model_idx: int, text: str) -> Dict[str, Any]:
        """Predict using a single model."""
        model = self.models[model_idx]
        processor = self.processors[model_idx]

        result = model.predict(text, processor)
        result["model_idx"] = model_idx
        return result

    def predict(self, text: str) -> Dict[str, Any]:
        """Predict using ensemble with parallel inference."""
        # Run predictions in parallel
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            futures = []
            for i in range(len(self.models)):
                future = executor.submit(self._predict_single, i, text)
                futures.append(future)

            # Collect results
            predictions = []
            for future in concurrent.futures.as_completed(futures):
                predictions.append(future.result())

        # Sort by model index for consistency
        predictions.sort(key=lambda x: x["model_idx"])

        # Apply voting strategy
        if self.voting_strategy == "majority":
            result = VotingStrategies.majority_vote(predictions)
        elif self.voting_strategy == "weighted":
            result = VotingStrategies.weighted_vote(predictions, self.weights)
        elif self.voting_strategy == "confidence":
            result = VotingStrategies.confidence_based(predictions)
        elif self.voting_strategy == "soft":
            result = VotingStrategies.soft_vote(predictions)
        else:
            raise ValueError(f"Unknown voting strategy: {self.voting_strategy}")

        # Add individual model predictions
        result["individual_predictions"] = predictions

        return result

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Predict for multiple texts."""
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str = None,
        voting_strategy: str = "majority",
        device: str = "cpu",
    ) -> "EnsembleDetector":
        """
        Load ensemble from pretrained models in directory.

        Looks for:
            - cnn_best.pt
            - lstm_best.pt
            - transformer_best.pt
        """
        if model_dir is None:
            # Use package models directory
            import os

            package_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            model_dir = os.path.join(package_dir, "models", "checkpoints")

        model_dir = Path(model_dir)

        # Create directory if it doesn't exist
        model_dir.mkdir(parents=True, exist_ok=True)

        model_configs = []
        expected_files = [
            ("cnn", "cnn_best.pt"),
            ("lstm", "lstm_best.pt"),
            ("transformer", "transformer_best.pt"),
        ]

        for model_type, filename in expected_files:
            model_path = model_dir / filename
            if model_path.exists():
                weight = 0.5 if model_type == "transformer" else 0.25
                model_configs.append(
                    {
                        "type": model_type,
                        "checkpoint_path": str(model_path),
                        "weight": weight,
                    }
                )

        if not model_configs:
            raise FileNotFoundError(
                f"No model checkpoints found in {model_dir}\n"
                f"Expected files: cnn_best.pt, lstm_best.pt, transformer_best.pt\n\n"
                f"To fix:\n"
                f"  1. Train individual models: promptscan train --model-type cnn|lstm|transformer\n"
                f"  2. Specify custom model directory with --model-dir\n"
                f"  3. Use single model instead of ensemble: --model-type cnn|lstm|transformer"
            )

        return cls(model_configs, voting_strategy, device)

    def get_model_info(self) -> List[Dict[str, Any]]:
        """Get information about loaded models."""
        info = []
        for i, (model, processor, weight, model_type) in enumerate(
            zip(self.models, self.processors, self.weights, self.model_types)
        ):
            model_info = {
                "index": i,
                "type": model_type,
                "parameters": sum(p.numel() for p in model.parameters()),
                "weight": weight,
                "device": str(model.get_device()),
            }
            info.append(model_info)
        return info
