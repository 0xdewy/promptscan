"""Prompt Detective - AI-powered prompt injection detection system."""

import os
from pathlib import Path

__version__ = VERSION = "0.1.1"


def get_model_path(model_name: str) -> Path:
    """
    Get the path to a model checkpoint file.

    Args:
        model_name: Name of the model file (e.g., "cnn_best.pt", "transformer_best.pt")

    Returns:
        Path to the model file

    Raises:
        FileNotFoundError: If model file is not found in any location
    """
    # First check if the file exists at the given path (for custom models)
    if os.path.exists(model_name):
        return Path(model_name)

    # Check environment variable for custom model directory
    env_model_dir = os.environ.get("PROMPTSCAN_MODEL_DIR")
    if env_model_dir:
        env_path = Path(env_model_dir) / model_name
        if env_path.exists():
            return env_path

    # Check in the package's checkpoints directory
    package_dir = Path(__file__).parent
    checkpoint_path = package_dir / "models" / "checkpoints" / model_name

    if checkpoint_path.exists():
        return checkpoint_path

    # Fallback to the old models/ directory (for backward compatibility)
    fallback_path = Path("models") / model_name
    if fallback_path.exists():
        return fallback_path

    # Try to find project root and check models directory
    # Look for pyproject.toml or setup.cfg to identify project root
    current = Path.cwd()
    project_root = None

    # Search up to 3 parent directories for project root
    for i in range(4):
        check_dir = current
        for _ in range(i):
            check_dir = check_dir.parent

        if (check_dir / "pyproject.toml").exists() or (
            check_dir / "setup.cfg"
        ).exists():
            project_root = check_dir
            break

    if project_root:
        project_model_path = project_root / "models" / model_name
        if project_model_path.exists():
            return project_model_path

    # If we get here, no model was found
    searched_paths = [
        f"1. Direct path: {model_name}",
        f"2. Package checkpoints: {checkpoint_path}",
        f"3. Local models directory: {fallback_path}",
    ]

    if env_model_dir:
        searched_paths.append(
            f"4. Environment variable PROMPTSCAN_MODEL_DIR: {env_path}"
        )

    if project_root:
        searched_paths.append(f"5. Project root models: {project_model_path}")

    raise FileNotFoundError(
        f"Model '{model_name}' not found.\n"
        f"Searched locations:\n" + "\n".join(searched_paths) + "\n\n"
        f"To fix:\n"
        f"  1. Train a model: promptscan train --model-type cnn|lstm|transformer\n"
        f"  2. Specify custom path with --model\n"
        f"  3. Check if models exist in expected locations\n"
        f"  4. Set PROMPTSCAN_MODEL_DIR environment variable"
    )


def get_default_model_save_path(model_name: str) -> Path:
    """
    Get the default path where a model should be saved.

    This is similar to get_model_path() but doesn't check if the file exists.
    Used by training commands to determine where to save new models.

    Args:
        model_name: Name of the model file (e.g., "cnn_best.pt", "transformer_best.pt")

    Returns:
        Path where the model should be saved
    """
    # First check if it looks like an absolute or relative path
    if os.path.isabs(model_name) or "/" in model_name or "\\" in model_name:
        return Path(model_name)

    # Default to saving in the local models directory
    return Path("models") / model_name


def get_default_model_paths() -> dict:
    """Get default paths for all model checkpoints."""
    return {
        "cnn": get_model_path("cnn_best.pt"),
        "lstm": get_model_path("lstm_best.pt"),
        "transformer": get_model_path("transformer_best.pt"),
        "best": get_model_path("best_model.pt"),
    }


# Import key components for easier access
try:
    from .unified_detector import UnifiedDetector
    from .feedback_store import ParquetFeedbackStore
    from .parquet_store import ParquetDataStore
except ImportError:
    # Allow partial imports for documentation generation
    pass

__all__ = [
    "__version__",
    "VERSION",
    "get_model_path",
    "get_default_model_save_path",
    "get_default_model_paths",
    "UnifiedDetector",
    "ParquetFeedbackStore",
    "ParquetDataStore",
]
