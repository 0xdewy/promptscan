#!/usr/bin/env python3
"""
Hugging Face Hub utilities for promptscan.
"""

import os
import shutil
from pathlib import Path
from typing import Optional

try:
    from huggingface_hub import hf_hub_download

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
            cache_dir = os.path.expanduser("~/.cache/promptscan")

    cache_path = Path(cache_dir) / "models" / model_name
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

    import logging as _logging
    import time as _time

    _logger = _logging.getLogger(__name__)

    for attempt in range(3):
        try:
            # Download safetensors file
            safetensors_path = hf_hub_download(
                repo_id=repo_id,
                filename=model_path_in_repo,
                cache_dir=str(cache_path),
                token=token,
                force_download=force_download,
                timeout=60,
            )

            # Download config file
            config_path = hf_hub_download(
                repo_id=repo_id,
                filename=config_path_in_repo,
                cache_dir=str(cache_path),
                token=token,
                force_download=force_download,
                timeout=60,
            )

            cache_safetensors = cache_path / f"{model_name}.safetensors"
            cache_config = cache_path / f"{model_name}.config.json"

            shutil.copy2(safetensors_path, cache_safetensors)
            shutil.copy2(config_path, cache_config)

            return cache_path / model_name

        except Exception as e:
            if attempt < 2:
                _logger.warning(
                    "Error downloading model from Hugging Face Hub "
                    "(attempt %d/3): %s", attempt + 1, e
                )
                _time.sleep(2 ** attempt)
            else:
                _logger.error(
                    "Failed to download model from Hugging Face Hub "
                    "after 3 attempts: %s", e
                )

    return None



