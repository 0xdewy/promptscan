---
language:
- en
license: mit
library_name: pytorch
tags:
- prompt-injection
- ai-safety
- security
- content-moderation
datasets:
- prompt-detective-dataset
metrics:
- accuracy
- f1
- precision
- recall
---

# Prompt Detective: Prompt Injection Detection Model

## Model Description

Prompt Detective is an ensemble-based prompt injection detection system that combines multiple neural network architectures for robust security analysis. The system analyzes text input and classifies it as either a safe prompt or a potential prompt injection attempt.

**Model Architecture**: Ensemble system combining CNN (for local pattern detection), LSTM (for sequential understanding), and Transformer (for contextual accuracy) models with configurable voting strategies.

**Intended Use**: This model is intended for use in AI safety pipelines to filter out malicious prompt injection attempts before they reach language models. The ensemble approach provides transparent voting with individual model confidence scores.

## Training Data

The model was trained on an aggregated dataset of:
- **6,362 safe prompts** (37.0%)
- **10,833 injection examples** (63.0%)
- **Total: 17,195 examples** (English and Spanish)

The dataset includes various types of prompt injection attacks:
- Direct instruction overrides
- Context manipulation
- Social engineering attempts
- Encoded/obfuscated injections
- Role-playing jailbreaks
- Multi-language injection attempts

## Usage

### Installation
```bash
# Using uv (recommended)
uv pip install prompt-detective

# Using pip
pip install prompt-detective
```

### Basic Usage
```python
from prompt_detective import UnifiedDetector

# Load the detector (ensemble is default)
detector = UnifiedDetector(model_type="ensemble")

# Analyze text for prompt injection
result = detector.predict("Ignore all previous instructions")
print(f"Result: {result['prediction']} ({result['confidence']:.2%})")

# Get individual model predictions in ensemble mode
if "individual_predictions" in result:
    for pred in result["individual_predictions"]:
        print(f"{pred.get('model_type', 'Unknown')}: {pred['prediction']} ({pred['confidence']:.2%})")
```

### Command Line Interface
```bash
# Analyze text (ensemble is default)
prompt-detective predict "Ignore all previous instructions"

# Analyze file
prompt-detective predict --file input.txt

# Analyze directory with summary
prompt-detective predict --dir ./prompts/ --summary

# Train a new model
prompt-detective train

# Import safe documentation from GitHub
prompt-detective insert --github https://github.com/python/cpython --label safe
```

## Performance

The ensemble model achieves the following performance metrics on the validation set:
- **Accuracy**: ~97%
- **F1 Score**: ~0.96
- **Precision**: ~0.95
- **Recall**: ~0.97

### Individual Model Performance
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

## Limitations

1. **Class Distribution**: The dataset has more injection examples than safe prompts (63% vs 37%)
2. **Evolving Threats**: New prompt injection techniques may not be covered
3. **False Positives**: Technical documentation may be incorrectly flagged (86.4% false positive rate on README.md files, 93.8% on model_card.md files)
4. **Context Limitations**: The model analyzes text in isolation without broader conversation context
5. **Language Coverage**: Better performance on English than Spanish prompts
6. **Documentation Bias**: The model has been trained to reduce false positives on technical documentation through batch import features

## Ethical Considerations

This model is designed to enhance AI safety by detecting malicious prompt injections. However, users should:

1. **Not rely solely on automated detection** - Human review is recommended for critical applications
2. **Consider false positives** - Legitimate but unusual prompts may be incorrectly flagged
3. **Regularly update the model** - As new injection techniques emerge, the model should be retrained
4. **Respect user privacy** - Only analyze text where appropriate consent has been obtained

## Training Details

- **Framework**: PyTorch
- **Training Time**: ~30 minutes per model on GPU
- **Batch Size**: 16 (reduced for memory safety)
- **Learning Rate**: 0.001
- **Optimizer**: AdamW
- **Loss Function**: Cross-Entropy
- **Data Source**: Automatically loads from `data/prompts.parquet` with fresh splits (80% train, 10% validation, 10% test)
- **Enhanced Training**: Supports batch import of safe documentation from GitHub repositories to reduce false positives

## Citation

If you use this model in your research, please cite:

```bibtex
@software{prompt_detective_2024,
  title = {Prompt Detective: Prompt Injection Detection Model},
  author = {Prompt Detective Contributors},
  year = {2024},
  url = {https://github.com/yourusername/prompt-detective}
}
```

## License

MIT License

## Contact

For questions, issues, or contributions, please open an issue on the [GitHub repository](https://github.com/yourusername/prompt-detective).