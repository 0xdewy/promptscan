#!/usr/bin/env python3
"""
Terminal color utilities for CLI output.
"""

import os
import sys


class Colors:
    """ANSI color codes with terminal detection."""

    # Basic colors
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"

    # Styles
    BOLD = "\033[1m"
    DIM = "\033[2m"
    UNDERLINE = "\033[4m"
    RESET = "\033[0m"

    # Backgrounds
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"

    @classmethod
    def supports_color(cls):
        """Check if terminal supports colors."""
        # Check environment variables
        if "NO_COLOR" in os.environ:
            return False
        if "FORCE_COLOR" in os.environ:
            return True

        # Check if stdout is a tty
        if not sys.stdout.isatty():
            return False

        # Platform-specific checks
        platform = sys.platform
        if platform == "win32":
            # Windows terminal color support
            return True

        return True

    @classmethod
    def colored(cls, text, color):
        """Return colored text if terminal supports it."""
        if cls.supports_color():
            return f"{color}{text}{cls.RESET}"
        return text

    @classmethod
    def prediction(cls, prediction, confidence=None):
        """Format prediction with appropriate color."""
        if prediction == "INJECTION":
            color = cls.RED
            icon = "🔴"
        else:  # SAFE
            color = cls.GREEN
            icon = "🟢"

        text = f"{icon} {prediction}"
        if confidence is not None:
            # Add confidence with color gradient
            conf_color = cls.confidence_color(confidence)
            conf_text = f"{confidence:.1%}"
            if cls.supports_color():
                text += f" ({conf_color}{conf_text}{cls.RESET})"
            else:
                text += f" ({conf_text})"

        return cls.colored(text, color)

    @classmethod
    def confidence_color(cls, confidence):
        """Get color code for confidence score."""
        if confidence >= 0.9:
            return cls.GREEN
        elif confidence >= 0.7:
            return cls.YELLOW
        elif confidence >= 0.5:
            return cls.YELLOW
        elif confidence >= 0.3:
            return cls.RED
        else:
            return cls.RED

    @classmethod
    def model_color(cls, model_idx):
        """Get color for model type based on index."""
        colors = [cls.BLUE, cls.MAGENTA, cls.CYAN]
        return colors[model_idx % len(colors)]

    @classmethod
    def header(cls, text):
        """Format header text."""
        return cls.colored(text, cls.BOLD)

    @classmethod
    def warning(cls, text):
        """Format warning text."""
        return cls.colored(text, cls.YELLOW)

    @classmethod
    def error(cls, text):
        """Format error text."""
        return cls.colored(text, cls.RED)

    @classmethod
    def success(cls, text):
        """Format success text."""
        return cls.colored(text, cls.GREEN)

    @classmethod
    def info(cls, text):
        """Format info text."""
        return cls.colored(text, cls.CYAN)
