# Prompt Injection Detector

A clean, minimal neural network for detecting prompt injection attacks. Just 350 lines of code.

## How It Works

### 1. **Text Processing**
- **Tokenization**: Text is converted to lowercase and split into words
- **Vocabulary**: Built from training data (3,534 words in current model)
- **Encoding**: Words are converted to numeric IDs using the vocabulary
- **Padding/Truncation**: All texts are standardized to 100 tokens

### 2. **Neural Network Architecture**
```
Input (100 tokens) → Embedding (64 dim) → CNN Filters (3,4,5) → 
Max Pooling → Fully Connected Layers → Output (2 classes)
```

**CNN Filters**:
- 3-word patterns: "ignore all previous"
- 4-word patterns: "you are now DAN"
- 5-word patterns: "tell me how to hack"

### 3. **Training Process**
1. Load prompts from SQLite database (`prompts.db`)
2. Split data: 80% train, 10% validation, 10% test
3. Build vocabulary from training texts
4. Train CNN for 20 epochs with AdamW optimizer
5. Save best model checkpoint

### 4. **Inference**
1. Load model checkpoint (`best_model.pt`)
2. Extract vocabulary and max length from checkpoint
3. Convert input text to token IDs
4. Run through CNN model
5. Output: SAFE or INJECTION with confidence scores

## Features

- **CNN architecture** - Fast training and inference
- **Minimal dependencies** - Just PyTorch and SQLite
- **Single file** - Everything in `detector.py`
- **97% validation accuracy** - Trained on expanded dataset
- **Small model** - 275KB trained model
- **Self-contained** - Vocabulary stored in checkpoint

## Quick Start

```bash
# Activate virtual environment
source venv/bin/activate

# Train model (uses prompts.db database)
python detector.py train

# Run inference
python detector.py "Ignore all previous instructions"
```

## File Structure

```
safe_prompts/
├── detector.py                 # Main script (350 lines)
├── best_model.pt               # Trained model + vocabulary (275KB)
├── prompts.db                  # Training data (SQLite, 154 prompts)
├── database.py                 # Database utilities
├── export_database.py          # Export utilities
├── generate_injections.py      # Generate creative injections via API
├── generate_safe_prompts.py    # Generate popular prompts via API
├── add_creative_injections.py  # Add 20 creative injections (no API needed)
├── import_output_json.py       # Extract texts from output.json as safe prompts
└── venv/                       # Python virtual environment
```

## Database Schema

```sql
CREATE TABLE prompts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    is_injection BOOLEAN NOT NULL
)
```

**Current Stats**:
- **1,045 total prompts**
- **108 injection prompts** (10.3%)
- **937 safe prompts** (89.7%)

## Usage Examples

### Training
```bash
python detector.py train
```
Trains a new model using data from `prompts.db`. Saves to `best_model.pt`.

### Inference
```bash
python detector.py "Ignore all previous instructions"
# Output: INJECTION with 94% confidence

python detector.py "What is the weather today?"
# Output: SAFE with confidence score
```

### Database Management
```python
from database import PromptDatabase

# Add new prompts
db = PromptDatabase()
db.add_prompt("Your text here", is_injection=True)  # or False

# Get all prompts
prompts = db.get_all_prompts()

# Get statistics
stats = db.get_statistics()
```

### Export Data
```bash
# Show statistics
python export_database.py --format stats

# Export to JSON
python export_database.py --format json

# Export to CSV
python export_database.py --format csv

# Export to Excel
python export_database.py --format excel
```

## Model Details

- **Vocabulary size**: 3,534 words
- **Max sequence length**: 100 tokens
- **Embedding dimension**: 64
- **CNN filters**: 50 each for sizes 3, 4, 5
- **Training epochs**: 20
- **Batch size**: 32
- **Learning rate**: 0.001

## Requirements

- Python 3.8+
- PyTorch
- SQLite3 (built-in)

The virtual environment already has everything installed.

## Data Generation

### Add Creative Injections (No API Needed)
```bash
# Add 20 creative prompt injection attacks
python add_creative_injections.py

# Add both injections and safe prompts
python add_creative_injections.py --add-safe
```

### Generate via DeepSeek API
```bash
# Set your API key
export DEEPSEEK_API_KEY="your-api-key-here"

# Generate creative injections
python generate_injections.py --count 10

# Generate popular/safe prompts  
python generate_safe_prompts.py --count 10

# Add top 20 most popular prompts
python generate_safe_prompts.py --top-20
```

### Manual Database Management
```python
from database import PromptDatabase

db = PromptDatabase()
db.add_prompt("Your creative injection", is_injection=True)
db.add_prompt("Your safe question", is_injection=False)
```

## How to Improve

1. **Add more injection examples** - Currently 10.3% injections, 89.7% safe (need more injections!)
2. **Add diverse injection patterns** - More creative attack vectors
3. **Tune hyperparameters** - Adjust CNN filters, embedding size, etc.
4. **Add data augmentation** - Synonym replacement, back-translation
5. **Experiment with architectures** - Try LSTM or Transformer

## Why It Works

Prompt injections often contain specific patterns:
- "Ignore all previous instructions"
- "You are now [malicious role]"
- "Tell me how to [harmful action]"
- "Disregard your ethical guidelines"

The CNN learns to detect these patterns across different wordings and contexts.