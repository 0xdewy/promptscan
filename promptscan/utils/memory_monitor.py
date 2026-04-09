#!/usr/bin/env python3
"""
Memory monitoring utilities to prevent crashes.
"""

import gc
import os
from typing import Dict, Optional

import torch

# Try to import psutil, but don't fail if not available
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None


class MemoryMonitor:
    """Monitor memory usage and prevent crashes."""

    def __init__(
        self, warning_threshold_mb: int = 8000, critical_threshold_mb: int = 10000
    ):
        """
        Initialize memory monitor.

        Args:
            warning_threshold_mb: Warn when memory exceeds this (MB)
            critical_threshold_mb: Take action when memory exceeds this (MB)
        """
        self.warning_threshold = warning_threshold_mb * 1024 * 1024  # Convert to bytes
        self.critical_threshold = critical_threshold_mb * 1024 * 1024

    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory usage information."""
        if not PSUTIL_AVAILABLE:
            return {"psutil_not_available": True}

        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()

            # System memory
            system_memory = psutil.virtual_memory()

            # PyTorch GPU memory if available
            torch_gpu_memory = {}
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / 1024**2
                    reserved = torch.cuda.memory_reserved(i) / 1024**2
                    torch_gpu_memory[f"gpu_{i}_allocated_mb"] = allocated
                    torch_gpu_memory[f"gpu_{i}_reserved_mb"] = reserved

            return {
                "process_rss_mb": memory_info.rss / 1024**2,
                "process_vms_mb": memory_info.vms / 1024**2,
                "system_available_mb": system_memory.available / 1024**2,
                "system_total_mb": system_memory.total / 1024**2,
                "system_percent_used": system_memory.percent,
                **torch_gpu_memory,
            }
        except Exception as e:
            return {"error": str(e)}

    def check_memory(self) -> Optional[str]:
        """
        Check memory usage and return warning message if thresholds exceeded.

        Returns:
            Warning message or None if memory is OK
        """
        info = self.get_memory_info()

        # Check if psutil is available
        if "psutil_not_available" in info:
            return "WARNING: psutil not available, memory monitoring disabled"
        if "error" in info:
            return f"WARNING: Memory monitoring error: {info['error']}"

        if info["process_rss_mb"] > (self.critical_threshold / 1024**2):
            return f"CRITICAL: Process memory {info['process_rss_mb']:.1f}MB exceeds critical threshold"
        elif info["process_rss_mb"] > (self.warning_threshold / 1024**2):
            return f"WARNING: Process memory {info['process_rss_mb']:.1f}MB exceeds warning threshold"

        if info["system_percent_used"] > 90:
            return f"CRITICAL: System memory {info['system_percent_used']:.1f}% used"
        elif info["system_percent_used"] > 80:
            return f"WARNING: System memory {info['system_percent_used']:.1f}% used"

        return None

    def force_garbage_collection(self):
        """Force garbage collection and clear PyTorch cache."""
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def log_memory_usage(self, prefix: str = ""):
        """Log current memory usage."""
        info = self.get_memory_info()

        # Check if psutil is available
        if "psutil_not_available" in info:
            message = f"{prefix}Memory monitoring disabled (psutil not installed)"
            print(message)
            return info
        if "error" in info:
            message = f"{prefix}Memory monitoring error: {info['error']}"
            print(message)
            return info

        message = f"{prefix}Memory: Process={info['process_rss_mb']:.1f}MB, "
        message += f"System={info['system_percent_used']:.1f}% used"

        if "gpu_0_allocated_mb" in info:
            message += f", GPU={info['gpu_0_allocated_mb']:.1f}MB"

        print(message)
        return info

    def safe_training_check(self, batch_size: int, dataset_size: int) -> bool:
        """
        Check if it's safe to train with given batch size and dataset.

        Returns:
            True if safe, False if memory might be insufficient
        """
        info = self.get_memory_info()

        # Check if psutil is available
        if "psutil_not_available" in info or "error" in info:
            print(
                "WARNING: Memory monitoring not available, using conservative defaults"
            )
            # Conservative default: don't use batch size > 32 without monitoring
            if batch_size > 32:
                print(
                    f"  Reducing batch size from {batch_size} to 32 (conservative default)"
                )
                return False
            return True

        # Estimate memory needed for batch
        # Rough estimate: 1MB per 100 samples in batch (very conservative)
        estimated_batch_memory_mb = batch_size * 0.01

        # Check if we have enough memory
        available_system_mb = info["system_available_mb"]

        if (
            estimated_batch_memory_mb > available_system_mb * 0.5
        ):  # Don't use more than 50% of available
            print(
                f"WARNING: Batch size {batch_size} might be too large for available memory"
            )
            print(f"  Estimated batch memory: {estimated_batch_memory_mb:.1f}MB")
            print(f"  Available system memory: {available_system_mb:.1f}MB")
            return False

        return True


def setup_memory_safe_training(batch_size: int = 32, max_memory_mb: int = 8000):
    """
    Setup memory-safe training environment.

    Args:
        batch_size: Initial batch size (may be reduced if memory is low)
        max_memory_mb: Maximum memory to use (MB)

    Returns:
        Adjusted batch size and memory monitor
    """
    monitor = MemoryMonitor(
        warning_threshold_mb=max_memory_mb * 0.7,
        critical_threshold_mb=max_memory_mb * 0.9,
    )

    # Log initial memory
    print("=" * 60)
    print("MEMORY SAFETY SETUP")
    print("=" * 60)
    monitor.log_memory_usage("Initial: ")

    # Check if batch size is safe
    safe = monitor.safe_training_check(batch_size, 10000)  # Assume 10k samples

    if not safe:
        # Reduce batch size
        new_batch_size = max(8, batch_size // 2)
        print(
            f"Reducing batch size from {batch_size} to {new_batch_size} for memory safety"
        )
        batch_size = new_batch_size

    # Force garbage collection
    monitor.force_garbage_collection()

    print("=" * 60)
    return batch_size, monitor


if __name__ == "__main__":
    # Test the memory monitor
    monitor = MemoryMonitor()
    info = monitor.log_memory_usage("Test: ")

    warning = monitor.check_memory()
    if warning:
        print(f"⚠️  {warning}")
    else:
        print("✅ Memory usage is within safe limits")
