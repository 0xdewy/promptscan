#!/usr/bin/env python3
"""
Device management utilities for PyTorch.
"""

import os
from contextlib import contextmanager
from typing import Optional

import torch


def get_device(device: Optional[str] = None) -> str:
    """
    Auto-detect best available device.

    Args:
        device: Device string ("cpu", "cuda", "auto", or None)

    Returns:
        Device string ("cpu" or "cuda")

    Examples:
        >>> get_device("auto")  # Returns "cuda" if available, else "cpu"
        >>> get_device("cpu")   # Returns "cpu"
        >>> get_device("cuda")  # Returns "cuda" if available, else "cpu"
    """
    if device is None or device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            # Set safer CUDA memory allocator settings
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        else:
            device = "cpu"
            # Limit CPU threads to prevent system freeze
            torch.set_num_threads(min(4, torch.get_num_threads()))

    if device == "cuda" and torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )
    else:
        if device == "cuda":
            print("CUDA requested but not available, falling back to CPU")
            device = "cpu"
        print(f"Using CPU with {torch.get_num_threads()} threads")

    return device


@contextmanager
def gpu_memory_context(max_memory_fraction: float = 0.8):
    """
    Context manager for GPU memory management.

    Args:
        max_memory_fraction: Maximum fraction of GPU memory to use (0.0 to 1.0)

    Yields:
        None

    Example:
        >>> with gpu_memory_context(max_memory_fraction=0.8):
        >>>     model = Model().cuda()
        >>>     # Training/inference code here
    """
    if torch.cuda.is_available():
        # Set memory fraction
        torch.cuda.set_per_process_memory_fraction(max_memory_fraction)

        # Clear cache at start
        torch.cuda.empty_cache()

        try:
            yield
        finally:
            # Clear cache at end
            torch.cuda.empty_cache()
    else:
        # No GPU available, just yield
        yield


def get_available_devices() -> dict:
    """
    Get information about available devices.

    Returns:
        Dictionary with device information
    """
    devices = {
        "cpu": {"available": True, "count": 1},
        "cuda": {"available": torch.cuda.is_available(), "count": 0},
    }

    if devices["cuda"]["available"]:
        devices["cuda"]["count"] = torch.cuda.device_count()
        devices["cuda"]["devices"] = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            devices["cuda"]["devices"].append(
                {
                    "name": torch.cuda.get_device_name(i),
                    "memory_gb": props.total_memory / 1024**3,
                    "capability": f"{props.major}.{props.minor}",
                }
            )

    return devices


def print_device_info(device: Optional[str] = None):
    """
    Print detailed information about available devices.

    Args:
        device: Optional device to print info for
    """
    devices = get_available_devices()

    print("=" * 60)
    print("Device Information")
    print("=" * 60)

    print("\nCPU:")
    print(f"  Available: {devices['cpu']['available']}")
    print(f"  Threads: {torch.get_num_threads()}")

    print("\nCUDA:")
    print(f"  Available: {devices['cuda']['available']}")
    if devices["cuda"]["available"]:
        print(f"  Devices: {devices['cuda']['count']}")
        for i, dev in enumerate(devices["cuda"]["devices"]):
            print(f"  Device {i}:")
            print(f"    Name: {dev['name']}")
            print(f"    Memory: {dev['memory_gb']:.1f} GB")
            print(f"    Compute Capability: {dev['capability']}")

    if device:
        selected = get_device(device)
        print(f"\nSelected device: {selected}")

    print("=" * 60)


if __name__ == "__main__":
    # Test the device utilities
    print_device_info("auto")

    # Test get_device
    print(f"\nget_device('auto'): {get_device('auto')}")
    print(f"get_device('cpu'): {get_device('cpu')}")
    print(f"get_device('cuda'): {get_device('cuda')}")

    # Test context manager
    print("\nTesting GPU memory context manager...")
    with gpu_memory_context(max_memory_fraction=0.8):
        print("Inside GPU memory context")
        if torch.cuda.is_available():
            x = torch.randn(1000, 1000).cuda()
            print(f"Created tensor on {x.device}")
