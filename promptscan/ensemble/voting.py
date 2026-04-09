#!/usr/bin/env python3
"""
Voting strategies for ensemble predictions.
"""

from collections import Counter, defaultdict
from typing import Any, Dict, List

import numpy as np


class VotingStrategies:
    """Collection of voting strategies for ensemble predictions."""

    @staticmethod
    def majority_vote(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Majority voting - each model gets one vote."""
        votes = Counter([p["prediction"] for p in predictions])
        winner, winner_count = votes.most_common(1)[0]

        # Average confidence for winning class
        winner_confs = [
            p["confidence"] for p in predictions if p["prediction"] == winner
        ]
        avg_confidence = sum(winner_confs) / len(winner_confs)

        # Calculate agreement ratio
        agreement = winner_count / len(predictions)

        return {
            "prediction": winner,
            "confidence": avg_confidence,
            "agreement": agreement,
            "votes": dict(votes),
            "strategy": "majority",
        }

    @staticmethod
    def weighted_vote(
        predictions: List[Dict[str, Any]], weights: List[float] = None
    ) -> Dict[str, Any]:
        """Weighted voting - models weighted by confidence or custom weights."""
        if weights is None:
            # Default: weight by model confidence
            weights = [p["confidence"] for p in predictions]

        # Normalize weights
        weights = np.array(weights) / sum(weights)

        # Weighted average of probabilities
        weighted_probs = defaultdict(float)
        for pred, weight in zip(predictions, weights):
            # Convert to probability distribution
            probs = {"SAFE": 0.0, "INJECTION": 0.0}
            if pred["prediction"] == "SAFE":
                probs["SAFE"] = pred["confidence"]
                probs["INJECTION"] = 1 - pred["confidence"]
            else:
                probs["INJECTION"] = pred["confidence"]
                probs["SAFE"] = 1 - pred["confidence"]

            for class_name, prob in probs.items():
                weighted_probs[class_name] += prob * weight

        # Select winner
        winner = max(weighted_probs.items(), key=lambda x: x[1])

        return {
            "prediction": winner[0],
            "confidence": winner[1],
            "probabilities": dict(weighted_probs),
            "strategy": "weighted",
        }

    @staticmethod
    def confidence_based(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select prediction with highest confidence."""
        best = max(predictions, key=lambda x: x["confidence"])

        # Count how many agree with the best
        agreement_count = sum(
            1 for p in predictions if p["prediction"] == best["prediction"]
        )
        agreement = agreement_count / len(predictions)

        return {
            "prediction": best["prediction"],
            "confidence": best["confidence"],
            "agreement": agreement,
            "source_model": best.get("model_type", "unknown"),
            "strategy": "confidence",
        }

    @staticmethod
    def soft_vote(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Soft voting - average probability distributions."""
        avg_probs = defaultdict(float)

        for pred in predictions:
            # Convert to probability distribution
            probs = {"SAFE": 0.0, "INJECTION": 0.0}
            if pred["prediction"] == "SAFE":
                probs["SAFE"] = pred["confidence"]
                probs["INJECTION"] = 1 - pred["confidence"]
            else:
                probs["INJECTION"] = pred["confidence"]
                probs["SAFE"] = 1 - pred["confidence"]

            for class_name, prob in probs.items():
                avg_probs[class_name] += prob

        # Average
        for class_name in avg_probs:
            avg_probs[class_name] /= len(predictions)

        # Select winner
        winner = max(avg_probs.items(), key=lambda x: x[1])

        return {
            "prediction": winner[0],
            "confidence": winner[1],
            "probabilities": dict(avg_probs),
            "strategy": "soft",
        }
