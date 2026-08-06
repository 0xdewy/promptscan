#!/usr/bin/env python3
"""
Ensemble detector with parallel inference and consensus voting.
"""

import concurrent.futures
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..models.cnn_model import SimpleCNN
from ..models.lstm_model import LSTMModel
from ..models.pretrained_model import PretrainedInjectionModel
from ..models.transformer_model import TransformerModel
from .voting import VotingStrategies


def _load_model_config(checkpoint_path: Optional[str]) -> Dict[str, Any]:
    """Load model metadata (best_threshold, calibration_temperature, val_f1) from config."""
    if checkpoint_path is None:
        return {}
    config_path = Path(checkpoint_path).with_suffix(".config.json")
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        return {
            "best_threshold": cfg.get("best_threshold", 0.5),
            "calibration_temperature": cfg.get("calibration_temperature", 1.0),
            "val_f1": cfg.get("val_f1", 0.5),
        }
    return {}


class EnsembleDetector:
    """Ensemble detector with multiple models and voting."""

    def __init__(
        self,
        model_configs: List[Dict[str, Any]],
        voting_strategy: str = "weighted",
        device: str = "cpu",
        max_workers: int = 4,
        confidence_threshold: float = 0.5,
    ):
        self.models = []
        self.processors = []
        self.weights = []
        self.model_types = []
        self.model_thresholds = []
        self.model_temperatures = []
        self.voting_strategy = voting_strategy
        self.device = device
        self.max_workers = max_workers
        self.confidence_threshold = confidence_threshold
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

        for config in model_configs:
            model_type = config["type"]
            checkpoint_path = config["checkpoint_path"]
            weight = config.get("weight", 1.0)

            meta = _load_model_config(checkpoint_path)
            threshold = config.get("confidence_threshold", meta.get("best_threshold", 0.5))
            temperature = config.get("calibration_temperature", meta.get("calibration_temperature", 1.0))

            if model_type == "cnn":
                model, processor = SimpleCNN.load(checkpoint_path, device)
            elif model_type == "lstm":
                model, processor = LSTMModel.load(checkpoint_path, device)
            elif model_type == "transformer":
                model, processor = TransformerModel.load(checkpoint_path, device)
            elif model_type == "pretrained":
                try:
                    model, processor = PretrainedInjectionModel.load(checkpoint_path, device)
                except Exception as e:
                    print(f"Warning: could not load pretrained model ({e}), skipping")
                    continue
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            self.models.append(model)
            self.processors.append(processor)
            self.weights.append(weight)
            self.model_types.append(model_type)
            self.model_thresholds.append(threshold)
            self.model_temperatures.append(temperature)

        print(f"Loaded {len(self.models)} models for ensemble detection")
        expected = len(model_configs)
        loaded = len(self.models)
        if loaded < expected:
            print(f"Warning: {expected - loaded} model(s) failed to load, ensemble using {loaded}/{expected} models")
        print(f"Models: {[c['type'] for c in model_configs]}")
        print(f"Voting strategy: {voting_strategy}")
        print(f"Confidence threshold: {confidence_threshold}")

    def _predict_single(self, model_idx: int, text: str) -> Dict[str, Any]:
        """Predict using a single model with its per-model threshold and temperature."""
        model = self.models[model_idx]
        processor = self.processors[model_idx]
        threshold = self.model_thresholds[model_idx]
        temperature = self.model_temperatures[model_idx]

        result = model.predict(
            text, processor,
            confidence_threshold=max(threshold, self.confidence_threshold),
            calibration_temperature=temperature,
        )
        result["model_idx"] = model_idx
        return result

    def predict(self, text: str) -> Dict[str, Any]:
        """Predict using ensemble with parallel inference."""
        futures = []
        for i in range(len(self.models)):
            future = self._executor.submit(self._predict_single, i, text)
            futures.append(future)

        predictions = []
        for future in concurrent.futures.as_completed(futures):
            try:
                predictions.append(future.result(timeout=30))
            except concurrent.futures.TimeoutError:
                for i, f in enumerate(futures):
                    if f is future:
                        print(f"Warning: model {self.model_types[i]} timed out during inference")
                        break
            except Exception as e:
                for i, f in enumerate(futures):
                    if f is future:
                        print(f"Warning: model {self.model_types[i]} failed during inference: {e}")
                        break

        predictions.sort(key=lambda x: x["model_idx"])

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

        result["individual_predictions"] = predictions
        return result

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Predict for multiple texts in parallel."""
        results = [None] * len(texts)
        futures_map = {}
        for i, text in enumerate(texts):
            future = self._executor.submit(self.predict, text)
            futures_map[future] = i

        for future in concurrent.futures.as_completed(futures_map):
            idx = futures_map[future]
            try:
                results[idx] = future.result(timeout=60)
            except concurrent.futures.TimeoutError:
                results[idx] = {
                    "prediction": "ERROR",
                    "confidence": 0.0,
                    "error": "timeout",
                }
            except Exception as e:
                results[idx] = {
                    "prediction": "ERROR",
                    "confidence": 0.0,
                    "error": str(e),
                }
        return results

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str = None,
        voting_strategy: str = "weighted",
        device: str = "cpu",
        confidence_threshold: float = 0.5,
    ) -> "EnsembleDetector":
        """
        Load ensemble from pretrained models with dynamic weights from configs.
        """
        from .. import get_model_path

        local_models = [
            ("cnn", "cnn_best"),
            ("lstm", "lstm_best"),
            ("transformer", "transformer_best"),
        ]

        model_configs = []
        total_f1 = 0.0
        temp_configs = []

        for model_type, model_name in local_models:
            try:
                checkpoint_path = get_model_path(model_name)
                meta = _load_model_config(str(checkpoint_path))
                f1 = meta.get("val_f1", 0.5)
                temp_configs.append({
                    "type": model_type,
                    "checkpoint_path": str(checkpoint_path),
                    "val_f1": f1,
                    "best_threshold": meta.get("best_threshold", 0.5),
                    "calibration_temperature": meta.get("calibration_temperature", 1.0),
                })
                total_f1 += f1
            except FileNotFoundError:
                print(f"Warning: {model_name} not found, skipping in ensemble")

        pretrained_f1 = 0.7
        total_f1 += pretrained_f1

        for tc in temp_configs:
            weight = tc["val_f1"] / total_f1 if total_f1 > 0 else 0.25
            model_configs.append({
                "type": tc["type"],
                "checkpoint_path": tc["checkpoint_path"],
                "weight": weight,
                "confidence_threshold": tc["best_threshold"],
                "calibration_temperature": tc["calibration_temperature"],
            })

        model_configs.append({
            "type": "pretrained",
            "checkpoint_path": None,
            "weight": pretrained_f1 / total_f1 if total_f1 > 0 else 0.25,
        })

        if not model_configs:
            raise FileNotFoundError(
                "No model checkpoints found for ensemble.\n"
                "Tried to load cnn_best, lstm_best, and transformer_best.\n"
                "Search order: local paths -> Hugging Face Hub auto-download.\n\n"
                "To fix:\n"
                "  1. Ensure huggingface-hub is installed: pip install huggingface-hub\n"
                "  2. Check network connectivity to huggingface.co\n"
                "  3. Or train individual models: ./train_all.sh --model-type cnn|lstm|transformer"
            )

        return cls(model_configs, voting_strategy, device,
                   confidence_threshold=confidence_threshold)

    def get_model_info(self) -> List[Dict[str, Any]]:
        """Get information about loaded models."""
        info = []
        for i, (model, _processor, weight, model_type) in enumerate(
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
