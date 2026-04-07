# Prompt Detective

**AI-powered prompt injection detection with transparent ensemble voting**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/promptscan.svg)](https://pypi.org/project/promptscan/)

Detect malicious prompt injection attacks with a production-ready ensemble of CNN, LSTM, and Transformer models. See exactly how each model votes with transparent confidence scores.

```bash
# Install and run in 30 seconds
uv pip install promptscan
promptscan predict "Ignore all previous instructions"
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
uv pip install promptscan

# Using pip
pip install promptscan
```

### Basic Usage

```bash
# Analyze text (ensemble is default)
promptscan predict "Ignore all previous instructions"

# Output shows individual model predictions:
# Individual model predictions:
#   - cnn: INJECTION (99.86%)
#   - lstm: SAFE (97.47%)
#   - transformer: INJECTION (99.85%)
# 
# Ensemble result: INJECTION (99.85%)

# Analyze files and directories
promptscan predict --file input.txt
promptscan predict --dir ./prompts/ --summary

# Use different model types
promptscan predict --model-type cnn "Test text"
promptscan predict --model-type lstm "Test text"
promptscan predict --model-type transformer "Test text"
```

## Installation Options

### From Source

```bash
git clone https://github.com/0xdewy/promptscan.git
cd promptscan
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
promptscan train

# Train specific model types
promptscan train --model-type lstm
promptscan train --model-type transformer

# Customize training parameters
promptscan train --epochs 10 --batch-size 32 --learning-rate 0.001
```

### Advanced Ensemble Options

```bash
# Use different voting strategies
promptscan predict --voting-strategy weighted "Test text"
promptscan predict --voting-strategy confidence "Test text"
promptscan predict --voting-strategy soft "Test text"

# Specify custom model directory
promptscan predict --model-dir ./my_models "Test text"

# Force CPU or GPU usage
promptscan predict --device cpu "Test text"
promptscan predict --device cuda "Test text"
```

### Data Management

```bash
# Export dataset statistics
promptscan export --format stats

# Export to JSON
promptscan export --format json --output prompts.json

# Export to CSV
promptscan export --format csv --output prompts.csv
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

## Batch Import and GitHub Integration

Prompt Detective now supports batch imports from multiple sources to enhance the dataset and address false positives.

### Importing Safe Documentation

The model sometimes flags legitimate documentation as prompt injections. To fix this, you can import safe documentation from various sources:

```bash
# Import from a GitHub repository
promptscan insert --github https://github.com/python/cpython --label safe

# Import from a local directory
promptscan insert --dir /path/to/docs --label safe --extensions .md,.txt

# Import specific files
promptscan insert --file README.md --file LICENSE.txt --label safe

# Use batch mode for non-interactive imports
promptscan insert --github https://github.com/nodejs/node --label safe --batch

# Preview without importing (dry run)
promptscan insert --github https://github.com/tensorflow/tensorflow --label safe --dry-run --verbose
```

### Advanced Import Options

```bash
# Filter by file extensions
promptscan insert --dir docs --label safe --extensions .md,.rst,.txt

# Exclude specific paths
promptscan insert --github https://github.com/org/repo --label safe --exclude "node_modules/" --exclude "test/"

# Limit file size
promptscan insert --dir docs --label safe --max-size 1MB

# Use GitHub token for higher rate limits
promptscan insert --github https://github.com/org/repo --label safe --github-token YOUR_TOKEN

# Or set environment variable
export GITHUB_TOKEN=your_token_here
promptscan insert --github https://github.com/org/repo --label safe
```

### Collecting Safe Documentation at Scale

Use the included script to collect safe documentation from curated repositories:

```bash
# Collect from multiple repositories (requires pandas)
python collect_safe_docs.py --max-repos 5

# Specify output file
python collect_safe_docs.py --output data/enhanced_dataset.parquet --max-repos 10
```

### Why Import Safe Documentation?

The original dataset has a bias that causes false positives:
- 86.4% of README.md files are incorrectly flagged as injections
- 93.8% of model_card.md files are incorrectly flagged
- Technical documentation uses instructional language that overlaps with injection patterns

By importing legitimate documentation as safe examples, you can:
1. Reduce false positive rates
2. Improve model accuracy on technical content
3. Create a more balanced dataset
4. Retrain the model for better performance

### Supported File Types

The batch importer extracts text from:
- **Markdown** (.md, .markdown)
- **Code files** (.py, .js, .ts, .java, .cpp, .c, .go, .rs) - extracts comments and docstrings
- **Documentation** (.rst, .adoc, .asciidoc)
- **Configuration** (.yaml, .yml, .json, .toml, .ini, .cfg)
- **Text files** (.txt, .text)
- **Web content** (.html, .htm, .xml) - extracts text from tags

### GitHub Rate Limits

- **Without token**: 60 requests/hour (quickly exhausted)
- **With token**: 5,000 requests/hour (recommended for large imports)
- **Get a token**: [GitHub Personal Access Tokens](https://github.com/settings/tokens)

## Training with Latest Data

The training command automatically loads data from `prompts.parquet` and creates fresh splits (80% train, 10% validation, 10% test), ensuring newly imported data is included in training.

### Data Architecture

- **`prompts.parquet`**: Single source of truth containing all prompts (19,649+ examples)
- **Dynamic splits**: Created fresh from `prompts.parquet` during training
- **Batch imports**: Add directly to `prompts.parquet` via GitHub/docs imports
- **Source files**: `injection_variations.parquet` and `safe_queries.parquet` are source files that have been imported
- **Legacy files**: `prompts_full.parquet`, `train.parquet`, `val.parquet`, `test.parquet` are deprecated and backed up in `data/backup/`

### Why This Matters

Previously, training used stale split files (`train.parquet`, `val.parquet`, `test.parquet`) that didn't include newly imported data. This caused:

- **86.4% false positive rate** on README.md files
- **93.8% false positive rate** on model_card.md files  
- Technical documentation incorrectly flagged as injections

### How It Works Now

```bash
# 1. Import safe documentation
promptscan insert --github https://github.com/python/cpython --label safe

# 2. Train model (automatically includes new data)
promptscan train --model-type ensemble

# 3. Test improved model
echo "Please follow these installation instructions" | promptscan predict
# Result: safe (not injection)
```

### Training Process

When you run `promptscan train`:

1. **Loads all data** from `data/prompts.parquet`
2. **Creates fresh splits** (80% train, 10% validation, 10% test)
3. **Includes new imports** (GitHub docs, local files, etc.)
4. **Trains on latest data** (no stale splits)

### Monitoring Data Growth

```bash
# Check dataset statistics
promptscan export --format stats

# Expected output after importing safe docs:
# Total prompts: 18,195 (was 17,195)
# Injection prompts: 10,833 (59.5%, was 63.0%)
# Safe prompts: 7,362 (40.5%, was 37.0%)
```

### Fixing False Positives

To reduce false positives on documentation:

1. **Import diverse safe documentation** from multiple repositories
2. **Retrain all models** to include new safe examples
3. **Test improvements** by checking documentation is no longer flagged

```bash
# Comprehensive fix workflow
promptscan insert --github https://github.com/python/cpython --label safe --batch
promptscan insert --github https://github.com/nodejs/node --label safe --batch
promptscan insert --github https://github.com/tensorflow/tensorflow --label safe --batch
promptscan train --model-type ensemble
```

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
├── utils/                   # Utility modules
├── github_client.py         # GitHub API client
├── text_extractors.py       # Text extraction from various file types
├── batch_importer.py        # Batch import framework
└── parquet_store.py         # Parquet data storage
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