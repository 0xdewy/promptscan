#!/usr/bin/env python3
"""
Hugging Face Hub utilities for promptscan.
"""

import os
import shutil
from pathlib import Path
from typing import Optional

try:
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from huggingface_hub.utils import HfHubHTTPError

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


def download_model_from_hf(
    repo_id: str = "0xdewy/promptscan",
    model_dir: str = "",
    model_name: str = "",
    token: Optional[str] = None,
    cache_dir: Optional[str] = None,
    force_download: bool = False,
) -> Optional[Path]:
    """
    Download a model from Hugging Face Hub.

    Args:
        repo_id: Hugging Face repository ID
        model_dir: Directory within the repository (e.g., "cnn", "lstm")
        model_name: Name of the model (e.g., "cnn_best")
        token: Hugging Face token (for private repos)
        cache_dir: Cache directory for downloaded files
        force_download: Force re-download even if cached

    Returns:
        Path to the downloaded model base file (without extension) or None if failed
    """
    if not HF_AVAILABLE:
        raise ImportError(
            "huggingface-hub is required to download models from Hugging Face Hub. "
            "Install with: pip install huggingface-hub"
        )

    if cache_dir is None:
        cache_dir = os.environ.get("PROMPTSCAN_CACHE_DIR")
        if not cache_dir:
            # Use HF cache by default
            cache_dir = os.environ.get(
                "HF_HOME", os.path.expanduser("~/.cache/huggingface")
            )

    # Create cache directory
    cache_path = Path(cache_dir) / "promptscan" / model_name
    cache_path.mkdir(parents=True, exist_ok=True)

    # Determine which files to download
    if model_dir:
        # Download from specific directory in repo
        safetensors_filename = "model.safetensors"
        config_filename = "config.json"
        model_path_in_repo = f"{model_dir}/{safetensors_filename}"
        config_path_in_repo = f"{model_dir}/{config_filename}"
    else:
        # Download from root of repo
        safetensors_filename = f"{model_name}.safetensors"
        config_filename = f"{model_name}.config.json"
        model_path_in_repo = safetensors_filename
        config_path_in_repo = config_filename

    try:
        # Download safetensors file
        safetensors_path = hf_hub_download(
            repo_id=repo_id,
            filename=model_path_in_repo,
            cache_dir=str(cache_path),
            token=token,
            force_download=force_download,
        )

        # Download config file
        config_path = hf_hub_download(
            repo_id=repo_id,
            filename=config_path_in_repo,
            cache_dir=str(cache_path),
            token=token,
            force_download=force_download,
        )

        # Copy to local models directory for easier access
        local_models_dir = Path("models")
        local_models_dir.mkdir(exist_ok=True)

        local_safetensors = local_models_dir / f"{model_name}.safetensors"
        local_config = local_models_dir / f"{model_name}.config.json"

        shutil.copy2(safetensors_path, local_safetensors)
        shutil.copy2(config_path, local_config)

        return local_models_dir / model_name

    except Exception as e:
        print(f"Error downloading model from Hugging Face Hub: {e}")
        return None


def download_all_models_from_hf(
    repo_id: str = "0xdewy/promptscan",
    output_dir: str = "models",
    token: Optional[str] = None,
    force_download: bool = False,
) -> bool:
    """
    Download all promptscan models from Hugging Face Hub.

    Args:
        repo_id: Hugging Face repository ID
        output_dir: Directory to save downloaded models
        token: Hugging Face token (for private repos)
        force_download: Force re-download even if cached

    Returns:
        True if all models downloaded successfully, False otherwise
    """
    if not HF_AVAILABLE:
        raise ImportError(
            "huggingface-hub is required to download models from Hugging Face Hub. "
            "Install with: pip install huggingface-hub"
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_types = ["cnn", "lstm", "transformer"]
    success_count = 0

    for model_type in model_types:
        print(f"Downloading {model_type.upper()} model...")

        try:
            # Download from model directory
            result = download_model_from_hf(
                repo_id=repo_id,
                model_dir=model_type,
                model_name=f"{model_type}_best",
                token=token,
                cache_dir=output_dir,
                force_download=force_download,
            )

            if result:
                print(f"  ✓ {model_type.upper()} model downloaded")
                success_count += 1
            else:
                print(f"  ✗ Failed to download {model_type} model")

        except Exception as e:
            print(f"  ✗ Error downloading {model_type} model: {e}")

    print(f"\nDownloaded {success_count}/{len(model_types)} models to {output_path}")
    return success_count == len(model_types)


def check_hf_model_available(
    repo_id: str = "0xdewy/promptscan",
    model_dir: str = "",
    token: Optional[str] = None,
) -> bool:
    """
    Check if a model is available on Hugging Face Hub.

    Args:
        repo_id: Hugging Face repository ID
        model_dir: Directory within the repository
        token: Hugging Face token (for private repos)

    Returns:
        True if model is available, False otherwise
    """
    if not HF_AVAILABLE:
        return False

    try:
        api = HfApi(token=token)

        if model_dir:
            # Check for model.safetensors in directory
            try:
                api.hf_hub_download(
                    repo_id=repo_id,
                    filename=f"{model_dir}/model.safetensors",
                    token=token,
                )
                return True
            except HfHubHTTPError:
                return False
        else:
            # Check if repo exists
            try:
                api.repo_info(repo_id=repo_id, repo_type="model")
                return True
            except HfHubHTTPError:
                return False

    except Exception:
        return False


def get_hf_model_info(
    repo_id: str = "0xdewy/promptscan",
    token: Optional[str] = None,
) -> Optional[dict]:
    """
    Get information about a Hugging Face model repository.

    Args:
        repo_id: Hugging Face repository ID
        token: Hugging Face token (for private repos)

    Returns:
        Dictionary with repository information or None if failed
    """
    if not HF_AVAILABLE:
        return None

    try:
        api = HfApi(token=token)
        repo_info = api.repo_info(repo_id=repo_id, repo_type="model")

        return {
            "id": repo_info.id,
            "last_modified": repo_info.last_modified,
            "tags": repo_info.tags,
            "private": repo_info.private,
            "downloads": repo_info.downloads,
            "likes": repo_info.likes,
        }
    except Exception:
        return None
