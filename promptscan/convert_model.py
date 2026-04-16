#!/usr/bin/env python3
"""
Convert old .pt model files to new safetensors format.
"""

import argparse
import json
import sys
from pathlib import Path

from .utils.colors import Colors

# Try to import torch and safetensors, but don't fail if not available
# (they'll fail later when actually used)
try:
    import torch
    from safetensors.torch import save_file

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    save_file = None


def convert_pt_to_safetensors(
    pt_path: str, output_path: str = None, force: bool = False
):
    """
    Convert a .pt model file to safetensors format.

    Args:
        pt_path: Path to the .pt file
        output_path: Output path (default: same name with .safetensors extension)
        force: Overwrite existing files
    """
    pt_path = Path(pt_path)

    if not pt_path.exists():
        print(f"{Colors.error(f'Error: File not found: {pt_path}')}")
        return False

    if not pt_path.suffix == ".pt":
        print(f"{Colors.error(f'Error: File must have .pt extension: {pt_path}')}")
        return False

    # Determine output paths
    if output_path is None:
        base_path = pt_path.parent / pt_path.stem
    else:
        base_path = Path(output_path).parent / Path(output_path).stem

    safetensors_path = base_path.with_suffix(".safetensors")
    config_path = base_path.with_suffix(".config.json")

    # Check if output files already exist
    if not force and (safetensors_path.exists() or config_path.exists()):
        print(f"{Colors.error('Error: Output files already exist:')}")
        print(f"  - {safetensors_path}")
        print(f"  - {config_path}")
        print(f"{Colors.warning('Use --force to overwrite.')}")
        return False

    if not TORCH_AVAILABLE:
        print(
            f"{Colors.error('Error: torch is not installed. Install with: pip install torch')}"
        )
        return False

    # Ensure torch is available in local scope
    import torch as local_torch
    from safetensors.torch import save_file as local_save_file

    print(f"Loading {pt_path}...")

    try:
        # First try pickle directly (old format)
        import pickle

        checkpoint = None
        loaded_with = None

        # First try pickle directly (old format)
        try:
            with open(pt_path, "rb") as f:
                checkpoint = pickle.load(f)
            loaded_with = "pickle"
            print("Loaded with pickle (old format)")
        except Exception:
            # Pickle failed, try torch.load
            pass

        # If pickle failed or checkpoint is not a dict, try torch.load
        if checkpoint is None or not isinstance(checkpoint, dict):
            try:
                checkpoint = local_torch.load(
                    pt_path, map_location="cpu", weights_only=True
                )
                loaded_with = "torch_weights_only"
                print("Loaded with torch.load (weights_only=True)")
            except Exception:
                # If that fails, try with safe globals for PyTorch 2.6+
                try:
                    import torch.serialization

                    with torch.serialization.safe_globals(
                        [local_torch.storage._load_from_bytes]
                    ):
                        print(
                            "Warning: Loading with safe_globals context - ensure file is from trusted source"
                        )
                        checkpoint = local_torch.load(
                            pt_path, map_location="cpu", weights_only=False
                        )
                        loaded_with = "torch_safe_globals"
                except Exception:
                    # Fall back to weights_only=False for compatibility
                    print(
                        "Warning: Loading with weights_only=False - ensure file is from trusted source"
                    )
                    checkpoint = local_torch.load(
                        pt_path, map_location="cpu", weights_only=False
                    )
                    loaded_with = "torch_unsafe"
                print("Loaded with torch.load (weights_only=True)")
            except Exception:
                # If that fails, try with safe globals for PyTorch 2.6+
                try:
                    import torch.serialization

                    with torch.serialization.safe_globals(
                        [local_torch.storage._load_from_bytes]
                    ):
                        print(
                            "Warning: Loading with safe_globals context - ensure file is from trusted source"
                        )
                        checkpoint = local_torch.load(
                            pt_path, map_location="cpu", weights_only=False
                        )
                except Exception:
                    # Fall back to weights_only=False for compatibility
                    print(
                        "Warning: Loading with weights_only=False - ensure file is from trusted source"
                    )
                    checkpoint = local_torch.load(
                        pt_path, map_location="cpu", weights_only=False
                    )

        # Extract model state dict
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            # Assume the file is already a state dict
            state_dict = checkpoint

        # Create config from checkpoint metadata
        config = {}

        # Copy known metadata fields
        metadata_fields = [
            "model_type",
            "vocab_size",
            "embedding_dim",
            "num_filters",
            "num_classes",
            "hidden_dim",
            "num_layers",
            "dropout",
            "model_name",
            "processor_config",
            "pytorch_version",
            "max_length",
            "min_freq",
            "vocab",
        ]

        for field in metadata_fields:
            if field in checkpoint:
                config[field] = checkpoint[field]

        # Add conversion metadata
        config["converted_from"] = str(pt_path)
        config["conversion_timestamp"] = local_torch.__version__

        # Save safetensors file
        print(f"Saving weights to {safetensors_path}...")
        local_save_file(state_dict, str(safetensors_path))

        # Save config file
        print(f"Saving config to {config_path}...")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print(
            f"{Colors.success(f'Successfully converted {pt_path} to safetensors format')}"
        )
        print(f"  {Colors.info(f'- Weights: {safetensors_path}')}")
        print(f"  {Colors.info(f'- Config: {config_path}')}")

        return True

    except Exception as e:
        print(f"{Colors.error(f'Error converting {pt_path}: {e}')}")
        return False


def convert_directory(pt_dir: str, output_dir: str = None, force: bool = False):
    """
    Convert all .pt files in a directory.

    Args:
        pt_dir: Directory containing .pt files
        output_dir: Output directory (default: same as input)
        force: Overwrite existing files
    """
    pt_dir = Path(pt_dir)

    if not pt_dir.exists():
        print(f"Error: Directory not found: {pt_dir}")
        return False

    if output_dir is None:
        output_dir = pt_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    pt_files = list(pt_dir.glob("*.pt"))

    if not pt_files:
        print(f"No .pt files found in {pt_dir}")
        return False

    print(f"Found {len(pt_files)} .pt files in {pt_dir}")

    success_count = 0
    for pt_file in pt_files:
        print(f"\nConverting {pt_file.name}...")

        if output_dir == pt_dir:
            output_path = None  # Use same directory
        else:
            output_path = output_dir / pt_file.stem

        if convert_pt_to_safetensors(pt_file, output_path, force):
            success_count += 1

    print(
        f"\nConversion complete: {success_count}/{len(pt_files)} files converted successfully"
    )
    return success_count > 0


def main():
    parser = argparse.ArgumentParser(
        description="Convert .pt model files to safetensors format"
    )
    parser.add_argument(
        "input", help="Input .pt file or directory containing .pt files"
    )
    parser.add_argument(
        "-o", "--output", help="Output file or directory (default: same as input)"
    )
    parser.add_argument(
        "-f", "--force", action="store_true", help="Overwrite existing files"
    )
    parser.add_argument(
        "--batch", action="store_true", help="Batch convert all .pt files in directory"
    )

    args = parser.parse_args()

    if args.batch or Path(args.input).is_dir():
        success = convert_directory(args.input, args.output, args.force)
    else:
        success = convert_pt_to_safetensors(args.input, args.output, args.force)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
