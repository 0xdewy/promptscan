"""Prompt Detective - AI-powered prompt injection detection system."""

import os
from pathlib import Path

__version__ = VERSION = "0.1.3"


def _is_source_install() -> bool:
    """Check if running from source tree (development) vs installed package."""
    current = Path.cwd()
    for i in range(4):
        check_dir = current
        for _ in range(i):
            check_dir = check_dir.parent
        if (check_dir / "pyproject.toml").exists() and (check_dir / ".git").exists():
            return True
    return False


def get_model_path(model_name: str, hf_fallback: bool = True) -> Path:
    """
    Get the path to a model checkpoint file (base name without extension).

    Args:
        model_name: Base name of the model (e.g., "cnn_best", "transformer_best")
                   or full path with .pt extension for backward compatibility.
        hf_fallback: If True, will attempt to download from Hugging Face Hub
            if not found locally.

    Returns:
        Path to the model base file (without extension)

    Raises:
        FileNotFoundError: If model files are not found in any location
    """
    from pathlib import Path

    # Remove .pt extension if present (for backward compatibility messages)
    if model_name.endswith(".pt"):
        model_name = model_name[:-3]

    # First check if the base path exists (for custom models)
    base_path = Path(model_name)

    # Check for safetensors + config files
    def check_model_files(path: Path) -> bool:
        safetensors_path = path.with_suffix(".safetensors")
        config_path = path.with_suffix(".config.json")
        return safetensors_path.exists() and config_path.exists()

    if check_model_files(base_path):
        return base_path

    # Check environment variable for custom model directory
    env_model_dir = os.environ.get("PROMPTSCAN_MODEL_DIR")
    if env_model_dir:
        env_path = Path(env_model_dir) / model_name
        if check_model_files(env_path):
            return env_path

    # Check local models/ directory first (user's retrained models)
    fallback_path = Path("models") / model_name
    if check_model_files(fallback_path):
        return fallback_path

    # Check in the package's checkpoints directory (only for installed package)
    package_dir = Path(__file__).parent
    checkpoint_path = package_dir / "models" / "checkpoints" / model_name

    if check_model_files(checkpoint_path):
        return checkpoint_path

    # Try to find project root and check models directory
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
        if check_model_files(project_model_path):
            return project_model_path

    # Check promptscan cache directory (~/.cache/promptscan/models/)
    cache_dir = os.environ.get(
        "PROMPTSCAN_CACHE_DIR",
        os.path.expanduser("~/.cache/promptscan"),
    )
    cache_model_path = Path(cache_dir) / "models" / model_name
    if check_model_files(cache_model_path):
        return cache_model_path

    # If we get here, no model was found locally
    # Try Hugging Face Hub as a fallback if enabled
    if (
        hf_fallback
        and not _is_source_install()
        and os.environ.get("PROMPTSCAN_USE_HF", "true").lower()
        in (
            "true",
            "1",
            "yes",
        )
    ):
        hf_model_map = {
            "cnn_best": ("cnn", "0xdewy/promptscan"),
            "lstm_best": ("lstm", "0xdewy/promptscan"),
            "transformer_best": ("transformer", "0xdewy/promptscan"),
        }

        if model_name in hf_model_map:
            model_dir, repo_id = hf_model_map[model_name]
            print(f"Downloading {model_name} from Hugging Face Hub...")
            try:
                from .hf_utils import download_model_from_hf

                local_path = download_model_from_hf(
                    repo_id=repo_id,
                    model_dir=model_dir,
                    model_name=model_name,
                    token=os.environ.get("HF_TOKEN"),
                )

                if local_path and check_model_files(local_path):
                    print(f"  -> Saved to {local_path}")
                    return local_path

            except ImportError as err:
                raise ImportError(
                    f"Model '{model_name}' not found locally and huggingface-hub "
                    "is required to download it. Install with: pip install huggingface-hub"
                ) from err
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download {model_name} from Hugging Face Hub: {e}"
                ) from e

    # If we get here, no model was found
    searched_paths = [
        f"1. Direct path: {base_path}.safetensors + {base_path}.config.json",
        f"2. Package checkpoints: {checkpoint_path}.safetensors + {checkpoint_path}.config.json",
        f"3. Local models directory: {fallback_path}.safetensors + {fallback_path}.config.json",
    ]

    if env_model_dir:
        searched_paths.append(
            f"4. Environment variable PROMPTSCAN_MODEL_DIR: {env_path}.safetensors + {env_path}.config.json"
        )

    if project_root:
        searched_paths.append(
            f"5. Project root models: {project_model_path}.safetensors + {project_model_path}.config.json"
        )

    searched_paths.append(
        f"6. PromptScan cache: {cache_model_path}.safetensors + {cache_model_path}.config.json"
    )

    if model_name in ["cnn_best", "lstm_best", "transformer_best"]:
        searched_paths.append(
            f"7. Hugging Face Hub: 0xdewy/promptscan/{model_name.split('_')[0]}/"
        )

    raise FileNotFoundError(
        f"Model '{model_name}' not found.\n"
        f"Searched locations (safetensors format):\n"
        + "\n".join(searched_paths)
        + "\n\n"
        "To fix:\n"
        "  1. Train a model: promptscan train --model-type cnn|lstm|transformer\n"
        "  2. Specify custom path with --model\n"
        "  3. Check if model files exist in expected locations\n"
        "  4. Set PROMPTSCAN_MODEL_DIR environment variable\n"
        "  5. Download from Hugging Face Hub: promptscan download-models"
    )


def get_default_model_save_path(model_name: str) -> Path:
    """
    Get the default path where a model should be saved (base name without extension).

    This is similar to get_model_path() but doesn't check if the file exists.
    Used by training commands to determine where to save new models.

    Args:
        model_name: Name of the model file (e.g., "cnn_best", "transformer_best")
                   or with .pt extension for backward compatibility.

    Returns:
        Path where the model should be saved (base name without extension)
    """
    # Remove .pt extension if present
    if model_name.endswith(".pt"):
        model_name = model_name[:-3]

    # First check if it looks like an absolute or relative path
    if os.path.isabs(model_name) or "/" in model_name or "\\" in model_name:
        return Path(model_name)

    # Default to saving in the local models directory
    return Path("models") / model_name


def get_default_model_paths() -> dict:
    """Get default paths for all model checkpoints."""
    return {
        "cnn": get_model_path("cnn_best"),
        "lstm": get_model_path("lstm_best"),
        "transformer": get_model_path("transformer_best"),
        "best": get_model_path("best_model"),
    }


# Import key components for easier access
try:
    from .feedback_store import ParquetFeedbackStore
    from .parquet_store import ParquetDataStore
    from .unified_detector import UnifiedDetector
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
