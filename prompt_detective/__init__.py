"""Prompt Detective - AI-powered prompt injection detection system."""

import os
from pathlib import Path

__version__ = VERSION = "0.1.0"


def get_model_path(model_name: str) -> Path:
    """
    Get the path to a model checkpoint file.

    Args:
        model_name: Name of the model file (e.g., "cnn_best.pt", "transformer_best.pt")

    Returns:
        Path to the model file
    """
    # First check if the file exists at the given path (for custom models)
    if os.path.exists(model_name):
        return Path(model_name)

    # Check in the package's checkpoints directory
    package_dir = Path(__file__).parent
    checkpoint_path = package_dir / "models" / "checkpoints" / model_name

    if checkpoint_path.exists():
        return checkpoint_path

    # Fallback to the old models/ directory (for backward compatibility)
    fallback_path = Path("models") / model_name
    return fallback_path


def get_default_model_paths() -> dict:
    """Get default paths for all model checkpoints."""
    return {
        "cnn": get_model_path("cnn_best.pt"),
        "lstm": get_model_path("lstm_best.pt"),
        "transformer": get_model_path("transformer_best.pt"),
        "best": get_model_path("best_model.pt"),
    }


__all__ = [
    "__version__",
    "VERSION",
    "get_model_path",
    "get_default_model_paths",
]
