"""promptscan: prompt injection detection."""

__version__ = VERSION = "0.2.2"

from .classifier import Classifier  # noqa: E402

__all__ = ["__version__", "VERSION", "Classifier"]
