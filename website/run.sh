#!/bin/bash

# Prompt Detective - Production Server
# Optimized for CPU deployment on Hetzner

set -e

echo "=========================================="
echo "PROMPT DETECTIVE - PRODUCTION SERVER"
echo "=========================================="
echo "Deployment: Hetzner CPU"
echo "Models: CNN, LSTM, Transformer ensemble"
echo "=========================================="
echo ""

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Script directory: $SCRIPT_DIR"
echo "Project root: $PROJECT_ROOT"

# Check if we're in the right place
if [ ! -f "$PROJECT_ROOT/pyproject.toml" ]; then
    echo "Error: Could not find project root"
    echo "Expected pyproject.toml at: $PROJECT_ROOT/pyproject.toml"
    exit 1
fi

# Change to project root
cd "$PROJECT_ROOT"
echo "Working directory: $(pwd)"

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python version: $PYTHON_VERSION"

# Parse version for comparison
MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")

if [[ $MAJOR -lt 3 ]] || ([[ $MAJOR -eq 3 ]] && [[ $MINOR -lt 8 ]]); then
    echo "Error: Python 3.8 or higher required"
    exit 1
fi

    # Check if dependencies are installed
    echo ""
    echo "Checking dependencies..."
    if ! uv run python -c "import torch, transformers, fastapi" 2>/dev/null; then
    echo "Installing production dependencies..."
    
    # Install with uv for better dependency management
    if command -v uv &> /dev/null; then
        echo "Using uv package manager..."
        # Install dependencies from requirements file
        echo "Installing dependencies..."
        uv pip install -r "$SCRIPT_DIR/requirements.txt"
    else
        echo "Using pip..."
        pip install -r "$SCRIPT_DIR/requirements.txt"
    fi
    
    echo "✓ Dependencies installed"
else
    echo "✓ All dependencies already installed"
fi

# Check if models are available
echo ""
echo "Checking model files..."
MODEL_DIR="$PROJECT_ROOT/models"
if [ -d "$MODEL_DIR" ]; then
    MODEL_COUNT=$(find "$MODEL_DIR" -name "*.pt" -o -name "*.pth" | wc -l)
    echo "✓ Found $MODEL_COUNT model files in $MODEL_DIR"
    
    # List available models
    echo "Available models:"
    find "$MODEL_DIR" -name "*.pt" -o -name "*.pth" | while read -r model; do
        echo "  - $(basename "$model")"
    done
else
    echo "⚠️  Warning: Model directory not found: $MODEL_DIR"
    echo "   The server will start but models may fail to load"
fi

# Set environment variables for production
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128  # Memory optimization for CPU
export OMP_NUM_THREADS=$(nproc)  # Use all CPU cores
export MKL_NUM_THREADS=$(nproc)

echo ""
echo "Environment configuration:"
echo "  PYTHONPATH: $PYTHONPATH"
echo "  CPU Threads: $(nproc)"
echo "  Memory: $(free -h | awk '/^Mem:/ {print $2}')"

# Start the server
echo ""
echo "=========================================="
echo "STARTING PRODUCTION SERVER"
echo "=========================================="
echo "Server: http://localhost:8000"
echo "API Docs: http://localhost:8000/api/docs"
echo "Health: http://localhost:8000/api/v1/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

    # Run the production server
    cd "$SCRIPT_DIR/api"
    uv run python main.py
