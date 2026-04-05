# Prompt Detective

**AI-powered prompt injection detection with transparent ensemble voting**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/prompt-detective.svg)](https://pypi.org/project/prompt-detective/)

Detect malicious prompt injection attacks with a production-ready ensemble of CNN, LSTM, and Transformer models. See exactly how each model votes with transparent confidence scores.

```bash
# Install and run in 30 seconds
uv pip install prompt-detective
prompt-detective predict "Ignore all previous instructions"
```

## Why Prompt Detective?

Prompt injections are emerging security threats where malicious users bypass AI system safeguards. Prompt Detective provides:

- **🔬 Multi-model ensemble** – Combines CNN speed, LSTM sequence understanding, and Transformer accuracy
- **📊 Transparent voting** – See individual model predictions and confidence scores
- **⚡ Production-ready** – Clean CLI, parallel inference, and self-contained models
- **📚 Comprehensive dataset** – 17,195 curated examples with 97% validation accuracy

## Features

- **Ensemble-first architecture** – Default mode combines CNN, LSTM, and Transformer models
- **Parallel inference** – All models run concurrently for ~50ms prediction time
- **Multiple voting strategies** – Majority (default), weighted, confidence-based, and soft voting
- **Flexible input** – Analyze text, files, directories, or URLs
- **Clean output** – Suppressed warnings and intuitive formatting
- **Self-contained** – No external API calls, vocabulary stored in checkpoints

## Quick Start

### Installation

```bash
# Using uv (recommended)
uv pip install prompt-detective

# Using pip
pip install prompt-detective
```

### Basic Usage

```bash
# Analyze text (ensemble is default)
prompt-detective predict "Ignore all previous instructions"

# Output shows individual model predictions:
# Individual model predictions:
#   - cnn: INJECTION (99.86%)
#   - lstm: SAFE (97.47%)
#   - transformer: INJECTION (99.85%)
# 
# Ensemble result: INJECTION (99.85%)

# Analyze files and directories
prompt-detective predict --file input.txt
prompt-detective predict --dir ./prompts/ --summary

# Use different model types
prompt-detective predict --model-type cnn "Test text"
prompt-detective predict --model-type lstm "Test text"
prompt-detective predict --model-type transformer "Test text"
```

## Installation Options

### From Source

```bash
git clone https://github.com/0xdewy/prompt-detective.git
cd prompt-detective
uv pip install -e .
```

### Development Setup

```bash
# Install with development dependencies
uv pip install -e ".[dev]"

# Run tests
uv run pytest tests/

# Format code
uv run black prompt_detective/ scripts/ tests/
uv run ruff check --fix prompt_detective/ scripts/ tests/
```

## Usage Examples

### Training Models

```bash
# Train CNN model (default for training)
prompt-detective train

# Train specific model types
prompt-detective train --model-type lstm
prompt-detective train --model-type transformer

# Customize training parameters
prompt-detective train --epochs 10 --batch-size 32 --learning-rate 0.001
```

### Advanced Ensemble Options

```bash
# Use different voting strategies
prompt-detective predict --voting-strategy weighted "Test text"
prompt-detective predict --voting-strategy confidence "Test text"
prompt-detective predict --voting-strategy soft "Test text"

# Specify custom model directory
prompt-detective predict --model-dir ./my_models "Test text"

# Force CPU or GPU usage
prompt-detective predict --device cpu "Test text"
prompt-detective predict --device cuda "Test text"
```

### Data Management

```bash
# Export dataset statistics
prompt-detective export --format stats

# Export to JSON
prompt-detective export --format json --output prompts.json

# Export to CSV
prompt-detective export --format csv --output prompts.csv
```

## Python API

```python
from prompt_detective import UnifiedDetector

# Load detector (ensemble is default)
detector = UnifiedDetector(model_type="ensemble")

# Analyze text
result = detector.predict("Ignore all previous instructions")
print(f"Result: {result['prediction']} ({result['confidence']:.2%})")

# Get individual model predictions in ensemble mode
if "individual_predictions" in result:
    for pred in result["individual_predictions"]:
        print(f"{pred.get('model_type', 'Unknown')}: {pred['prediction']} ({pred['confidence']:.2%})")
```

## Architecture

### Ensemble System

```
┌─────────────────────────────────────────────────────────┐
│                    Input Text                           │
└─────────────────┬─────────────────┬─────────────────────┘
                  │                 │
          ┌───────▼──────┐  ┌───────▼──────┐  ┌─────────────┐
          │     CNN      │  │     LSTM     │  │ Transformer │
          │   (Fast)     │  │ (Sequential) │  │ (Accurate)  │
          └───────┬──────┘  └───────┬──────┘  └──────┬──────┘
                  │                 │                 │
          ┌───────▼─────────────────▼─────────────────▼──────┐
          │              Voting Strategy                      │
          │        (Majority/Weighted/Confidence/Soft)        │
          └───────────────────────┬───────────────────────────┘
                                  │
                          ┌───────▼──────┐
                          │  Final       │
                          │  Prediction  │
                          └──────────────┘
```

### Model Details

| Model | Architecture | Parameters | Strength | Inference Time |
|-------|-------------|------------|----------|----------------|
| **CNN** | Convolutional Neural Network | 2.7M | Local pattern detection | ~10ms |
| **LSTM** | Bidirectional LSTM | 3.3M | Sequential understanding | ~15ms |
| **Transformer** | DistilBERT fine-tuned | 67M | Contextual accuracy | ~25ms |
| **Ensemble** | All three models | 73M | Combined robustness | ~50ms |

### Voting Strategies

- **Majority** (default): Each model gets one vote
- **Weighted**: Models weighted by confidence or custom weights
- **Confidence**: Select prediction with highest confidence
- **Soft**: Average probability distributions

## Dataset

Prompt Detective is trained on a comprehensive dataset of 17,195 examples:

- **10,833 injection prompts** (63.0%)
- **6,362 safe prompts** (37.0%)
- **Multilingual**: English (primary) and Spanish (secondary)
- **Sources**: Curated from multiple security research projects
- **Split**: 80% train, 10% validation, 10% test
- **Accuracy**: 97% validation accuracy on ensemble

## Development

### Project Structure

```
prompt_detective/
├── __init__.py              # Package initialization
├── cli.py                   # Command-line interface
├── unified_detector.py      # Unified detector interface
├── ensemble/                # Ensemble detection system
├── models/                  # CNN, LSTM, Transformer implementations
├── processors/              # Text processors
├── training/                # Training framework
└── utils/                   # Utility modules
```

### Building and Publishing

```bash
# Build package
uv build

# Build wheel only
uv build --wheel

# Publish to PyPI
uv publish
```

## Requirements

- Python 3.8+
- PyTorch 2.0.0+
- pandas 2.0.0+
- transformers 4.30.0+ (for Transformer model)
- tokenizers 0.13.0+ (for Transformer model)

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Acknowledgments

- Dataset contributors: `deepset/prompt-injections`, `AnaBelenBarbero/detect-prompt-injection`
- Model architectures: PyTorch, Hugging Face Transformers
- Tooling: uv, black, ruff, pytest

---

**Prompt Detective** – Transparent, robust prompt injection detection for AI security.