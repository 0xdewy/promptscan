# PromptScan

**AI-powered prompt injection detection with transparent ensemble voting**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/promptscan.svg)](https://pypi.org/project/promptscan/)

Detect malicious prompt injection attacks with a production-ready ensemble of CNN, LSTM, and Transformer models. See exactly how each model votes with transparent confidence scores.

```bash
# Install and run in 30 seconds
pip install promptscan
promptscan predict "Ignore all previous instructions"
```

## Why PromptScan?

Prompt injections are emerging security threats where malicious users bypass AI system safeguards. PromptScan provides:

- **🔬 Multi-model ensemble** – Combines CNN speed with LSTM sequence understanding and Transformer context
- **🔒 Safetensors format** – Secure model storage without pickle security risks  
- **☁️ Hugging Face Hub** – Automatic model download when not found locally
- **📊 Transparent voting** – See individual model predictions and confidence scores
- **⚡ Production-ready** – Clean CLI, parallel inference, and self-contained models

## Features

- **Models**: CNN (2.7M), LSTM (3.3M), Transformer (67M), Ensemble (all three)
- **Format**: Safetensors + JSON config (no pickle `.pt` files in distribution)
- **Distribution**: PyPI package + Hugging Face Hub repository
- **CLI**: `predict`, `train`, `hf download/upload/list`, `convert-model`
- **API**: `UnifiedDetector` with Python interface
- **Voting strategies**: Majority (default), weighted, confidence-based, soft

## Quick Start

### Installation

```bash
# Using pip (recommended)
pip install promptscan

# Using uv
uv pip install promptscan

# From source
git clone https://github.com/0xdewy/promptscan.git
cd promptscan
pip install -e .
```

### Basic Usage

```bash
# Analyze text (ensemble is default)
promptscan predict "Ignore all previous instructions"

# Output shows individual model predictions:
# Individual model predictions:
#   - cnn: INJECTION (99.86%)
#   - lstm: SAFE (97.47%)
#   - transformer: INJECTION (95.21%)
# 
# Ensemble result: INJECTION (99.86%)

# Use specific model types
promptscan predict --model-type cnn "Test text"
promptscan predict --model-type lstm "Test text"
promptscan predict --model-type transformer "Test text"

# Analyze files and directories
promptscan predict --file input.txt
promptscan predict --dir ./prompts/ --summary
```

### Hugging Face Hub Integration

```bash
# Download models from Hugging Face Hub
promptscan hf download --model cnn
promptscan hf download --model all

# List available models
promptscan hf list

# Upload models (requires HF_TOKEN)
promptscan hf upload --model-dir ./models/
```

## Architecture

### Ensemble System

```
┌─────────────────────────────────────────────────────────┐
│                    Input Text                           │
└─────────────────┬─────────────────┬─────────────────────┘
                  │                 │                 │
          ┌───────▼──────┐  ┌───────▼──────┐  ┌──────▼──────┐
          │     CNN      │  │     LSTM     │  │ Transformer │
          │   (Fast)     │  │ (Sequential) │  │ (Contextual)│
          └───────┬──────┘  └───────┬──────┘  └──────┬──────┘
                  │                 │                 │
          ┌───────▼─────────────────▼─────────────────▼──────┐
          │        Voting Strategy         │
          │ (Majority/Weighted/Confidence) │
          └─────────────────┬──────────────┘
                            │
                    ┌───────▼──────┐
                    │  Final       │
                    │  Prediction  │
                    └──────────────┘
```

### Model Specifications

| Model | Parameters | Architecture | Strength | Format |
|-------|------------|--------------|----------|--------|
| **CNN** | 2.7M | Convolutional Neural Network | Local pattern detection | Safetensors |
| **LSTM** | 3.3M | Bidirectional LSTM | Sequential understanding | Safetensors |
| **Transformer** | 67M | Transformer encoder | Contextual understanding | Safetensors |
| **Ensemble** | All | Combined voting | Robustness across patterns | Multiple |

**Note**: All models use safetensors format for security. Training checkpoints may use `.pt` format internally.

## Python API

```python
from promptscan import UnifiedDetector

# Load detector (ensemble is default)
detector = UnifiedDetector(model_type="ensemble")

# Analyze text
result = detector.predict("Ignore all previous instructions")
print(f"Result: {result['prediction']} ({result['confidence']:.2%})")

# Get individual model predictions in ensemble mode
if "individual_predictions" in result:
    for pred in result["individual_predictions"]:
        model_type = pred.get('model_type', 'Unknown')
        print(f"{model_type}: {pred['prediction']} ({pred['confidence']:.2%})")
```

## Dataset

PromptScan is trained on a curated dataset of 17,195 examples:

- **10,833 injection prompts** (63.0%)
- **6,362 safe prompts** (37.0%)
- **Multilingual**: English (primary) with Spanish examples
- **Sources**: Curated from security research projects
- **Accuracy**: High validation accuracy on ensemble

## Development

### Project Structure

```
promptscan/
├── __init__.py              # Package initialization, version, model paths
├── cli.py                   # Command-line interface
├── unified_detector.py      # Unified detector interface
├── models/                  # CNN, LSTM, Transformer implementations
├── ensemble/                # Ensemble detection system
├── hf_utils.py             # Hugging Face Hub integration
├── processors/              # Text processors
├── training/                # Training framework
└── utils/                   # Utility modules
```

### Development Setup

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black promptscan/ scripts/ tests/
ruff check --fix promptscan/

# Type checking
mypy promptscan/
```

### Building and Publishing

```bash
# Build package
python -m build

# Check package
twine check dist/*

# Publish to PyPI (requires token)
twine upload dist/*

# Or using uv
uv publish dist/promptscan-*.whl
```

## Requirements

- Python 3.8+
- PyTorch 2.0.0+
- safetensors>=0.4.0
- huggingface-hub>=0.20.0
- pandas>=2.0.0
- transformers>=4.30.0 (for Transformer model)

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

---

**PromptScan** – Transparent, robust prompt injection detection for AI security.