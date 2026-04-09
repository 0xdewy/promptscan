#!/bin/bash
# Simple wrapper script to run 

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if virtual environment exists
if [ ! -d "$SCRIPT_DIR/.venv" ]; then
    echo "Error: Virtual environment not found at $SCRIPT_DIR/.venv"
    echo "Please run: uv sync"
    exit 1
fi

# Activate virtual environment and run the command
source "$SCRIPT_DIR/.venv/bin/activate"

uv run python -m promptscan.cli
