#!/usr/bin/env python3
"""
Voting strategies for ensemble predictions.

Voting strategies now use native class_probabilities from model predictions
instead of reconstructing pseudo-probabilities from hard predictions.
"""

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

import numpy as np


AGREEMENT_GATE = 0.6


class VotingStrategies:
    """Collection of voting strategies for ensemble predictions."""

    @staticmethod
    def _get_class_probs(pred: Dict[str, Any]) -> Dict[str, float]:
        """Get class probabilities from a prediction result."""
        if "class_probabilities" in pred:
            return pred["class_probabilities"]
        confidence = pred.get("confidence", 0.5)
        if pred.get("prediction") == "INJECTION":
            return {"SAFE": 1.0 - confidence, "INJECTION": confidence}
        return {"SAFE": confidence, "INJECTION": 1.0 - confidence}

    @staticmethod
    def majority_vote(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Majority voting - each model gets one vote."""
        votes = Counter([p["prediction"] for p in predictions])
        winner, winner_count = votes.most_common(1)[0]
        agreement = winner_count / len(predictions)

        avg_probs = defaultdict(float)
        for p in predictions:
            probs = VotingStrategies._get_class_probs(p)
            for cls_name, prob in probs.items():
                avg_probs[cls_name] += prob
        for cls_name in avg_probs:
            avg_probs[cls_name] /= len(predictions)

        veredict = winner if agreement >= AGREEMENT_GATE else "uncertain"
        confidence = avg_probs[winner] if agreement >= AGREEMENT_GATE else min(avg_probs[winner], 0.5)

        return {
            "prediction": veredict,
            "confidence": confidence,
            "agreement": agreement,
            "votes": dict(votes),
            "probabilities": dict(avg_probs),
            "strategy": "majority",
        }

    @staticmethod
    def weighted_vote(
        predictions: List[Dict[str, Any]], weights: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """Weighted voting using native class probabilities."""
        if weights is None:
            weights = [p.get("confidence", 0.5) for p in predictions]

        weights = np.array(weights) / sum(weights)

        weighted_probs = defaultdict(float)
        for pred, weight in zip(predictions, weights):
            probs = VotingStrategies._get_class_probs(pred)
            for cls_name, prob in probs.items():
                weighted_probs[cls_name] += prob * weight

        winner = max(weighted_probs.items(), key=lambda x: x[1])

        vote_counts = Counter([p["prediction"] for p in predictions])
        winner_votes = vote_counts.get(winner[0], 0)
        agreement = winner_votes / len(predictions) if predictions else 0.0

        verdict = winner[0] if agreement >= AGREEMENT_GATE else "uncertain"
        confidence = (
            weighted_probs[winner[0]] if agreement >= AGREEMENT_GATE
            else min(weighted_probs[winner[0]], 0.5)
        )

        return {
            "prediction": verdict,
            "confidence": confidence,
            "agreement": agreement,
            "probabilities": dict(weighted_probs),
            "strategy": "weighted",
        }

    @staticmethod
    def confidence_based(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select prediction with highest confidence."""
        best = max(predictions, key=lambda x: x.get("confidence", 0))

        agreement_count = sum(
            1 for p in predictions if p["prediction"] == best["prediction"]
        )
        agreement = agreement_count / len(predictions) if predictions else 0.0

        verdict = best["prediction"] if agreement >= AGREEMENT_GATE else "uncertain"
        confidence = (
            best["confidence"] if agreement >= AGREEMENT_GATE
            else min(best["confidence"], 0.5)
        )

        return {
            "prediction": verdict,
            "confidence": confidence,
            "agreement": agreement,
            "source_model": best.get("model_type", "unknown"),
            "strategy": "confidence",
        }

    @staticmethod
    def soft_vote(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Soft voting - average native probability distributions."""
        avg_probs = defaultdict(float)

        for pred in predictions:
            probs = VotingStrategies._get_class_probs(pred)
            for cls_name, prob in probs.items():
                avg_probs[cls_name] += prob

        for cls_name in avg_probs:
            avg_probs[cls_name] /= len(predictions) if predictions else 1

        winner = max(avg_probs.items(), key=lambda x: x[1])

        vote_counts = Counter([p["prediction"] for p in predictions])
        winner_votes = vote_counts.get(winner[0], 0)
        agreement = winner_votes / len(predictions) if predictions else 0.0

        verdict = winner[0] if agreement >= AGREEMENT_GATE else "uncertain"
        confidence = (
            avg_probs[winner[0]] if agreement >= AGREEMENT_GATE
            else min(avg_probs[winner[0]], 0.5)
        )

        return {
            "prediction": verdict,
            "confidence": confidence,
            "agreement": agreement,
            "probabilities": dict(avg_probs),
            "strategy": "soft",
        }
