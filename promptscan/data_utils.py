"""
Data access utilities for Safe Prompts package.
Uses importlib.resources for accessing package data files.
"""

import shutil
import sys
import warnings
from pathlib import Path
from typing import Optional, Tuple


def get_package_data_dir() -> Optional[Path]:
    """Get the package data directory if available."""
    # Try multiple locations where package data might be installed

    # 1. Try system data directory (for installed packages)
    import site

    # Check common data directories
    data_dirs = []

    # System prefix directories
    prefixes = [sys.prefix, sys.base_prefix]
    for prefix in prefixes:
        if prefix:
            data_dirs.extend(
                [
                    Path(prefix) / "share" / "promptscan",
                    Path(prefix) / "local" / "share" / "promptscan",
                ]
            )

    # User site directories
    try:
        for site_dir in site.getsitepackages():
            data_dirs.append(Path(site_dir).parent / "share" / "promptscan")
    except Exception:
        pass

    # Check each directory
    for data_dir in data_dirs:
        if data_dir.exists():
            # Check if it contains data files
            parquet_files = list(data_dir.glob("*.parquet"))
            if parquet_files:
                return data_dir

    # 2. Try development directory (for editable installs)
    try:
        import promptscan

        package_dir = Path(promptscan.__file__).parent
        # Check parent directory for data
        project_root = package_dir.parent.parent
        data_dir = project_root / "data"
        if data_dir.exists() and list(data_dir.glob("*.parquet")):
            return data_dir
    except (ImportError, AttributeError):
        pass

    return None


def get_package_models_dir() -> Optional[Path]:
    """Get the package models directory if available."""
    try:
        import importlib.resources

        with importlib.resources.path("promptscan", "models") as models_dir:
            if models_dir.exists():
                return models_dir
    except (ImportError, FileNotFoundError):
        pass
    return None


def ensure_data_files(dest_dir: Optional[Path] = None) -> Tuple[Path, Path, Path]:
    """
    Ensure data files are available locally.

    NOTE: This function is deprecated. The training pipeline now uses
    prompts.parquet directly and creates dynamic splits.

    Args:
        dest_dir: Destination directory (default: current directory/data)

    Returns:
        Tuple of (train_path, val_path, test_path) - placeholder paths for backward compatibility
    """
    import warnings

    warnings.warn(
        "ensure_data_files is deprecated. Training now uses prompts.parquet with dynamic splits.",
        DeprecationWarning,
        stacklevel=2,
    )

    if dest_dir is None:
        dest_dir = Path.cwd() / "data"

    dest_dir.mkdir(parents=True, exist_ok=True)

    # Data files to copy (for backward compatibility)
    # Note: Static split files are deprecated, using prompts.parquet instead
    data_files = ["prompts.parquet"]
    dest_paths = []

    package_data_dir = get_package_data_dir()

    for filename in data_files:
        dest_path = dest_dir / filename

        # Check if file already exists
        if dest_path.exists():
            dest_paths.append(dest_path)
            continue

        # Try to copy from package data
        if package_data_dir:
            source_path = package_data_dir / filename
            if source_path.exists():
                shutil.copy2(source_path, dest_path)
                print(f"Copied {filename} to {dest_path}")
                dest_paths.append(dest_path)
                continue

        # File not available
        warnings.warn(
            f"Data file {filename} not found in package. "
            f"Creating empty placeholder at {dest_path}"
        )

        # Create empty placeholder
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq

        empty_df = pd.DataFrame(
            {"id": [], "text": [], "is_injection": [], "source": [], "text_length": []}
        )
        table = pa.Table.from_pandas(empty_df)
        pq.write_table(table, dest_path)
        dest_paths.append(dest_path)

        # File not available
        warnings.warn(
            f"Data file {filename} not found in package. "
            f"You may need to run the migration script first.",
            RuntimeWarning,
            stacklevel=2,
        )
        dest_paths.append(dest_path)

    return tuple(dest_paths)


def ensure_model_file(dest_dir: Optional[Path] = None, model_type: str = "cnn") -> Path:
    """
    Ensure model file is available locally.

    Args:
        dest_dir: Destination directory (default: current directory/models)
        model_type: Type of model ("cnn", "lstm", "transformer")

    Returns:
        Path to model file
    """
    if dest_dir is None:
        dest_dir = Path.cwd() / "models"

    dest_dir.mkdir(parents=True, exist_ok=True)

    # Determine filename based on model type
    if model_type == "cnn":
        filename = "best_model.pt"
    elif model_type == "lstm":
        filename = "lstm_best.pt"
    elif model_type == "transformer":
        filename = "transformer_best.pt"
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    dest_path = dest_dir / filename

    # Check if model already exists
    if dest_path.exists():
        return dest_path

    # Try to copy from package
    package_models_dir = get_package_models_dir()
    if package_models_dir:
        source_path = package_models_dir / filename
        if source_path.exists():
            shutil.copy2(source_path, dest_path)
            print(f"Copied {model_type} model to {dest_path}")
            return dest_path

    # Model not available
    warnings.warn(
        f"Pre-trained {model_type} model not found in package. "
        f"You may need to train a model first with 'promptscan train --model-type {model_type}'.",
        RuntimeWarning,
        stacklevel=2,
    )
    return dest_path


def get_default_data_paths() -> Tuple[Path, Path, Path]:
    """
    Get default data paths, using package data if available.

    NOTE: This function is deprecated. The training pipeline now uses
    prompts.parquet directly and creates dynamic splits.

    Returns:
        Tuple of (train_path, val_path, test_path) - placeholder paths for backward compatibility
    """
    import warnings

    warnings.warn(
        "get_default_data_paths is deprecated. Training now uses prompts.parquet with dynamic splits.",
        DeprecationWarning,
        stacklevel=2,
    )

    # First try package data
    package_data_dir = get_package_data_dir()
    if package_data_dir:
        prompts_path = package_data_dir / "prompts.parquet"
        if prompts_path.exists():
            # Return the same path three times for backward compatibility
            # Callers should use prompts.parquet with dynamic splits
            return prompts_path, prompts_path, prompts_path

    # Fall back to local data directory
    data_dir = Path.cwd() / "data"
    prompts_path = data_dir / "prompts.parquet"
    return prompts_path, prompts_path, prompts_path


def get_default_model_path() -> Path:
    """
    Get default model path, using package model if available.

    Returns:
        Path to model file
    """
    # First try package model
    package_models_dir = get_package_models_dir()
    if package_models_dir:
        model_path = package_models_dir / "best_model.pt"
        if model_path.exists():
            return model_path

    # Fall back to local models directory
    return Path.cwd() / "models" / "best_model.pt"


def list_available_data() -> dict:
    """List available data files in package."""
    result = {"data": [], "models": []}

    package_data_dir = get_package_data_dir()
    if package_data_dir:
        for file in package_data_dir.glob("*.parquet"):
            result["data"].append(file.name)

    package_models_dir = get_package_models_dir()
    if package_models_dir:
        for file in package_models_dir.glob("*.pt"):
            result["models"].append(file.name)

    return result


if __name__ == "__main__":
    # Test the utilities
    print("Testing data utilities...")
    print(f"Package data dir: {get_package_data_dir()}")
    print(f"Package models dir: {get_package_models_dir()}")
    print(f"Available data: {list_available_data()}")

    # Try to ensure files
    try:
        train, val, test = ensure_data_files()
        print("\nData paths:")
        print(f"  Train: {train}")
        print(f"  Val: {val}")
        print(f"  Test: {test}")

        model_path = ensure_model_file()
        print(f"Model path: {model_path}")
    except Exception as e:
        print(f"Error: {e}")
