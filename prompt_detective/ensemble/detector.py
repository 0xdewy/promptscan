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
        model_dir: str = "models",
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
        model_dir = Path(model_dir)
        model_configs = []

        # CNN model
        cnn_path = model_dir / "cnn_best.pt"
        if cnn_path.exists():
            model_configs.append(
                {"type": "cnn", "checkpoint_path": str(cnn_path), "weight": 0.25}
            )

        # LSTM model
        lstm_path = model_dir / "lstm_best.pt"
        if lstm_path.exists():
            model_configs.append(
                {"type": "lstm", "checkpoint_path": str(lstm_path), "weight": 0.25}
            )

        # Transformer model
        transformer_path = model_dir / "transformer_best.pt"
        if transformer_path.exists():
            model_configs.append(
                {
                    "type": "transformer",
                    "checkpoint_path": str(transformer_path),
                    "weight": 0.5,
                }
            )

        if not model_configs:
            raise FileNotFoundError(f"No model checkpoints found in {model_dir}")

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
