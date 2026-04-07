---
language:
- en
- es
task_categories:
- text-classification
task_ids:
- prompt-injection-detection
- ai-safety
size_categories:
- 10K<n<100K
license: mit
multilinguality:
- multilingual
source_datasets:
- original
- extended
pretty_name: Prompt Detective Dataset
dataset_info:
  features:
  - name: text
    dtype: string
  - name: is_injection
    dtype: bool
  splits:
  - name: train
    num_bytes: 13916774
    num_examples: 13756
  - name: validation
    num_bytes: 1658880
    num_examples: 1719
  - name: test
    num_bytes: 1708032
    num_examples: 1720
  configs:
  - config_name: default
    data_files:
    - split: train
      path: train.parquet
    - split: validation
      path: val.parquet
    - split: test
      path: test.parquet
  - config_name: dynamic
    data_files:
    - split: full
      path: prompts.parquet
    note: Modern training uses prompts.parquet with dynamic splits (80% train, 10% validation, 10% test)
tags:
- prompt-injection
- ai-safety
- security
- content-moderation
---

# Prompt Detective Dataset

## Dataset Description

A comprehensive dataset of 17,195 text prompts labeled for prompt injection detection. The dataset contains examples of both safe prompts and various types of prompt injection attacks in English and Spanish.

### Dataset Summary

- **Total Examples**: 17,195 (base dataset)
- **Safe Prompts**: 6,362 (37.0%)
- **Injection Examples**: 10,833 (63.0%)
- **Average Text Length**: ~500 characters
- **Languages**: English, Spanish
- **Data Splits**: Train (13,756), Validation (1,719), Test (1,720)

### Enhanced Dataset Features
The dataset can be enhanced through batch import features:
- **GitHub Integration**: Import safe documentation from repositories to reduce false positives
- **Dynamic Splits**: Training automatically creates fresh splits from `data/prompts.parquet`
- **False Positive Reduction**: Importing technical documentation addresses 86.4% false positive rate on README.md files and 93.8% on model_card.md files

### Supported Tasks

- **Prompt Injection Detection**: Binary classification task to identify whether a text prompt contains a prompt injection attack.

### Languages

The dataset contains text in English (primary) and Spanish (secondary). All examples are labeled for prompt injection detection.

## Dataset Structure

### Data Fields

- `text`: The text prompt (string)
- `is_injection`: Boolean label indicating whether the text is a prompt injection (True) or safe prompt (False)

### Data Splits

The dataset is provided as three splits:
- **Train**: 13,756 examples (80%)
- **Validation**: 1,719 examples (10%)
- **Test**: 1,720 examples (10%)

## Dataset Creation

### Curation Rationale

The dataset was created to train models for detecting prompt injection attacks in AI systems. Prompt injection is a security vulnerability where malicious users craft inputs that cause AI systems to bypass safety guidelines.

### Source Data

The dataset was created through aggregation and curation of multiple sources:

1. **Original Dataset** (10,737 examples):
   - Manually collected prompt injection examples
   - Generated safe prompts covering various topics
   - Synthetic injection examples using pattern-based approaches

2. **External Sources** (6,458 examples):
   - `deepset/prompt-injections` (Apache 2.0 License): 662 examples of prompt injections and benign prompts
   - `AnaBelenBarbero/detect-prompt-injection`: Processed English and Spanish examples from contrasto.ai project
   - Additional curated examples from security research

### Data Processing

1. **Deduplication**: Removed duplicate text entries across all sources
2. **Language Filtering**: Kept English and Spanish examples only
3. **Label Standardization**: Converted all labels to boolean `is_injection` format
4. **Quality Filtering**: Removed empty or malformed entries

### Annotations

#### Annotation process
Examples were labeled through multiple methods:
1. **Manual annotation**: Original examples reviewed by AI safety researchers
2. **Source labels**: External datasets with pre-existing labels
3. **Automated filtering**: Pattern-based identification of injection attempts

#### Who are the annotators?
- Prompt Detective project team (original dataset)
- deepset team (`deepset/prompt-injections` dataset)
- contrasto.ai team (`AnaBelenBarbero/detect-prompt-injection` project)

### Personal and Sensitive Information

The dataset contains synthetic, curated, and publicly available examples. No real user data or personally identifiable information is included.

## Considerations for Using the Data

### Social Impact of Dataset

This dataset supports the development of AI safety tools that can help prevent malicious use of language models and protect AI systems from prompt injection attacks.

### Discussion of Biases

The dataset has the following characteristics:
- **Class distribution**: 63% injections, 37% safe (improved balance from original)
- **Language bias**: Primarily English with some Spanish examples
- **Source bias**: Mix of synthetic, generated, and real-world examples
- **Style patterns**: Injection examples include various attack patterns (direct overrides, role-playing, encoded instructions)
- **Documentation bias**: Technical documentation may be incorrectly flagged as injections (addressed through batch import features)
- **Import capabilities**: Can be enhanced with safe documentation from GitHub repositories to improve balance

### Other Known Limitations

1. **Evolving threats**: New prompt injection techniques may not be represented
2. **Context limitations**: Examples are analyzed in isolation without conversation history
3. **False positive risk**: Some legitimate but unusual prompts may resemble injections
4. **Language coverage**: Limited Spanish examples compared to English
5. **Temporal coverage**: Examples reflect injection techniques up to 2024

## Additional Information

### Dataset Curators

- Prompt Detective project team (primary curation)
- Contributors from aggregated datasets

### Licensing Information

**MIT License**

**Note on aggregated data**: This dataset includes examples from:
- `deepset/prompt-injections` (Apache 2.0 License)
- `AnaBelenBarbero/detect-prompt-injection` (no explicit license, used with attribution)
- Original Prompt Detective dataset (MIT License)

All aggregated data is used in accordance with source licenses and with proper attribution.

### Citation Information

```bibtex
@dataset{prompt_detective_dataset_2024,
  title = {Prompt Detective Dataset (Extended)},
  author = {Prompt Detective Contributors and Dataset Aggregators},
  year = {2024},
  url = {https://github.com/yourusername/prompt-detective},
  note = {Aggregated dataset including examples from deepset/prompt-injections and AnaBelenBarbero/detect-prompt-injection}
}
```

### Contributions

Thanks to all contributors who helped create and curate this dataset.