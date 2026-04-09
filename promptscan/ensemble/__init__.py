"""
Ensemble detection system with consensus voting.
"""

from .detector import EnsembleDetector
from .voting import VotingStrategies

__all__ = ["EnsembleDetector", "VotingStrategies"]
