# Agent Guidelines for PromptScan

## Overview

PromptScan detects prompt injection attacks using an ensemble of CNN, LSTM, and Transformer models. Key architectural decisions:

- **Safetensors format only** for distributed models (no pickle `.pt` files)
- **Hugging Face Hub** as primary distribution channel
- **Fallback loading**: local → HF Hub automatic download
- **Ensemble-first design** with transparent voting
- **PyPI distribution** with minimal package size

## Code Style & Conventions

### Imports
```python
# Standard library
import os
from pathlib import Path
from typing import Dict, List, Optional

# Third-party
import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

# Local modules
from .base_model import BaseModel
from .utils.device import get_device
```

### Type Hints
- Use type hints for all function signatures
- Return `Optional[T]` for functions that can return `None`
- Use `Dict[str, Any]` for flexible return dictionaries
- Import types from `typing` module

### Docstrings
```python
def predict(self, text: str, processor) -> Dict[str, Any]:
    """
    Predict if text contains prompt injection.

    Args:
        text: Input text to analyze
        processor: Text processor for tokenization

    Returns:
        Dictionary with prediction, confidence, and metadata

    Raises:
        ValueError: If text is empty or invalid
    """
```

### Error Handling
- Raise specific exceptions (`FileNotFoundError`, `ValueError`, `ImportError`)
- Catch exceptions at appropriate levels
- Provide helpful error messages with suggestions

## Architecture

### Core Components
```
promptscan/
├── __init__.py           # Package init, version, get_model_path()
├── unified_detector.py   # Main interface (UnifiedDetector)
├── models/               # Model implementations
│   ├── base_model.py     # Abstract BaseModel class
│   ├── cnn_model.py      # SimpleCNN
│   ├── lstm_model.py     # LSTMModel  
│   └── transformer_model.py
├── ensemble/             # Ensemble voting system
├── cli.py               # Command-line interface (58657 lines)
├── hf_utils.py          # Hugging Face Hub integration
├── convert_model.py     # .pt to .safetensors conversion
└── utils/               # Device, text processing, memory monitoring
```

### Model System

**BaseModel Abstract Class** (`promptscan/models/base_model.py`):
- `load()`: Load model from safetensors + config
- `save()`: Save model as safetensors + config
- `predict()`: Make prediction on text
- `load_from_hf()`: Download from Hugging Face Hub

**Model Format**:
- Weights: `.safetensors` file
- Config: `.config.json` file (architecture, vocab, metadata)
- **Never** commit `.pt` pickle files to repository
- Training checkpoints can use `.pt` internally (not distributed)

**Loading Logic** (`promptscan/__init__.py:get_model_path()`):
1. Check local path
2. Check package `models/checkpoints/` directory
3. Fallback to Hugging Face Hub download
4. Raise `FileNotFoundError` if all fail

### Hugging Face Integration

**Repository**: `0xdewy/promptscan`
**Token**: `HF_TOKEN` environment variable required for uploads
**Structure**:
```
0xdewy/promptscan/
├── cnn/
│   ├── model.safetensors
│   └── config.json
├── lstm/
│   ├── model.safetensors
│   └── config.json
└── transformer/
    ├── model.safetensors
    └── config.json
```

**Utilities** (`promptscan/hf_utils.py`):
- `download_model_from_hf()`: Download model from HF Hub
- `upload_model_to_hf()`: Upload model to HF Hub
- `check_model_on_hf()`: Check if model exists on HF Hub

## CLI Structure

**Main Commands** (`promptscan.cli:main()`):
- `predict`: Detect prompt injections
- `train`: Train models
- `hf`: Hugging Face operations
- `convert-model`: Convert `.pt` to `.safetensors`

**Command Patterns**:
```python
def predict_command(args):
    """Handle predict command."""
    detector = UnifiedDetector(
        model_type=args.model_type,
        model_path=args.model,
        device=args.device,
        voting_strategy=args.voting_strategy,
    )
    result = detector.predict(args.text)
    _display_prediction(result, args.model_type, detector)
```

**HF Subcommands**:
- `hf download`: Download models from HF Hub
- `hf upload`: Upload models to HF Hub  
- `hf list`: List available models on HF Hub

## Testing

### Test Commands
```bash
# Run all tests
uv run pytest tests/

# Run specific test file
uv run pytest tests/test_detector.py

# Run with coverage
uv run pytest --cov=promptscan tests/

# Type checking
uv run mypy promptscan/

# Linting
uv run ruff check --fix promptscan/
```

### Test Structure
- `tests/test_detector.py`: Core functionality tests
- `tests/fixtures/`: Test data files
- Tests should verify both local and HF Hub loading
- Mock HF Hub API calls for unit tests

## Build & Publish

### Package Building
```bash
# Clean previous builds
rm -rf dist/ build/ promptscan.egg-info/

# Build package
uv build
# or
python -m build

# Verify package
twine check dist/*
```

### PyPI Publishing
```bash
# Publish using uv (requires token)
uv publish dist/promptscan-*.whl
# Enter username: __token__
# Enter password: [PyPI API token]

# Or using twine
twine upload dist/*
```

**Version Bumping**:
1. Update `version` in `pyproject.toml`
2. Update `__version__` in `promptscan/__init__.py`
3. Rebuild package
4. Publish to PyPI

### Hugging Face Publishing
```bash
# Upload all models
python upload_to_hf.py

# Or use CLI
promptscan hf upload --model-dir ./models/
```

**Requirements**:
- `HF_TOKEN` environment variable set
- Models must be in safetensors format
- Config files must be present

## Common Agent Tasks

### 1. Add New Model Type
```python
# 1. Create new model class extending BaseModel
class NewModel(BaseModel):
    def __init__(self, config):
        super().__init__()
        # Architecture definition
    
    def forward(self, inputs):
        # Implementation
    
    @classmethod
    def load(cls, checkpoint_path, device="cpu"):
        # Load from safetensors + config

# 2. Add to unified_detector.py
if model_type == "newmodel":
    model_path = str(get_model_path("newmodel_best"))
    self.detector = NewModel.load(model_path, device=self.device)

# 3. Add CLI support in cli.py
parser_predict.add_argument("--model-type", choices=["cnn", "lstm", "transformer", "ensemble", "newmodel"])

# 4. Add test cases
```

### 2. Fix Model Loading Bug
```bash
# 1. Write failing test
uv run pytest tests/test_models.py -xvs -k "test_load"

# 2. Debug loading issue
python -c "from promptscan.models.cnn_model import SimpleCNN; m = SimpleCNN.load('models/cnn_best')"

# 3. Fix in base_model.py or specific model class
# 4. Verify fix
uv run pytest tests/test_models.py -xvs -k "test_load"
```

### 3. Update CLI Command
```python
# 1. Add argument parser
subparsers = parser.add_subparsers(dest="command")
parser_new = subparsers.add_parser("new-command", help="New command help")

# 2. Add arguments
parser_new.add_argument("--option", help="Option help")

# 3. Implement command function
def new_command(args):
    """Handle new command."""
    # Implementation

# 4. Register in command dispatch
if args.command == "new-command":
    new_command(args)
```

### 4. Convert .pt to Safetensors
```bash
# Convert single model
promptscan convert-model models/cnn_best.pt models/cnn_best

# Convert all models
for model in cnn lstm transformer; do
    promptscan convert-model models/${model}_best.pt models/${model}_best
done

# Verify conversion
ls -la models/*.safetensors models/*.config.json
```

## Important Notes

### Security
- **Never** load untrusted `.pt` pickle files
- Always use `safetensors` for distributed models
- Training checkpoints (`.pt`) are for internal use only

### Dependencies
- Core: `torch`, `safetensors`, `huggingface-hub`
- Optional: `transformers`, `tokenizers` (for Transformer model)
- Dev: `pytest`, `black`, `ruff`, `mypy`, `build`, `twine`

### Environment Variables
- `HF_TOKEN`: Hugging Face authentication token
- `PROMPTSCAN_CACHE_DIR`: Custom cache directory
- `TRANSFORMERS_VERBOSITY`: Set to "error" to suppress warnings

### File Locations
- **Package models**: `promptscan/models/checkpoints/` (small demo models)
- **Large models**: `models/` directory (250MB+ each)
- **Data**: `data/prompts.parquet` (main dataset)
- **Build artifacts**: `dist/`, `build/`, `promptscan.egg-info/`

## Troubleshooting

### PyPI Publishing Issues
```bash
# Check if version already exists
curl -s "https://pypi.org/pypi/promptscan/json" | jq -r '.info.version'

# Clean and rebuild
rm -rf dist/ build/ promptscan.egg-info/
uv build

# Try different upload method
twine upload dist/*
```

### Hugging Face Upload Issues
```bash
# Check token
echo $HF_TOKEN

# Test HF CLI
huggingface-cli whoami

# Upload manually
python upload_to_hf.py --verbose
```

### Model Loading Issues
```bash
# Check file existence
ls -la models/*.safetensors

# Test loading
python -c "from safetensors.torch import load_file; data = load_file('models/cnn_best.safetensors')"

# Check config
cat models/cnn_best.config.json | jq .
```

## Quick Reference

```bash
# Development
uv pip install -e ".[dev]"
uv run pytest
uv run black promptscan/
uv run ruff check --fix promptscan/

# Building
uv build
twine check dist/*

# Publishing
uv publish dist/promptscan-*.whl
python upload_to_hf.py

# Testing in clean environment
python -m venv test-env
source test-env/bin/activate
pip install promptscan
promptscan --version
```