# Data Validation Report

Generated from `data/merged.parquet`

## Summary Statistics

- **Total Rows**: 483,386
- **Total Columns**: 11
- **Unique Sources**: 8,130

## Null Value Analysis

| Column | Null Count | Null % |
|--------|------------|--------|
| source | 16 | 0.0% |
| text_length | 466,630 | 96.53% |
| review_date | 483,371 | 100.0% |
| original_prediction | 483,371 | 100.0% |
| original_user_label | 483,371 | 100.0% |
| original_confidence | 483,371 | 100.0% |
| original_source | 483,371 | 100.0% |
| original_timestamp | 483,371 | 100.0% |

## Text Quality Issues

- **Empty texts**: 0
- **Very short texts (< 5 chars)**: 6
- **Duplicate texts**: 0

## Label Distribution

- **is_injection=True**: 114,987 (23.79%)
- **is_injection=False**: 368,399 (76.21%)

### Injection Rate by Source (Top 15)

| Source | Injections | Total | Rate |
|--------|------------|-------|------|
| imoxto_prompt_injection_cleaned_dataset-v2 | 93842 | 436697 | 21.5% |
| S-Labs_prompt-injection-dataset | 4782 | 11050 | 43.3% |
| Octavio-Santana_prompt-injection-attack-detection- | 3037 | 6312 | 48.1% |
| prompts.db | 4735 | 5274 | 89.8% |
| adfksfasbjsdk_Prompt-injection-dataset | 2595 | 4335 | 59.9% |
| prompts.json | 1210 | 4157 | 29.1% |
| injection_en.csv | 2554 | 2554 | 100.0% |
| injection_spanish.csv | 2168 | 2168 | 100.0% |
| benign_spanish.csv | 0 | 1753 | 0.0% |
| benign_en.csv | 0 | 838 | 0.0% |
| cgoosen_prompt_injection_ctf_dataset_2 | 38 | 77 | 49.4% |
| user_feedback_reviewed | 9 | 15 | 60.0% |
| prompts | 6 | 12 | 50.0% |
| file:README.md | 0 | 3 | 0.0% |
| file:LICENSE | 0 | 3 | 0.0% |

## Source Analysis

- **Total unique sources**: 8130
- **Major sources (>= 1000 rows)**: 9
- **Suspicious sources (flagged for removal)**: 8110

### Suspicious Sources Detected:
- `adfksfasbjsdk_Prompt-injection-dataset`
- `prompts.json`
- `prompts.db`
- `injection_en.csv`
- `injection_spanish.csv`
- `benign_spanish.csv`
- `benign_en.csv`
- `file:.idea/.gitignore`
- `file:.idea/chatgpt_system_prompt.iml`
- `file:.scripts/README.md`
- `file:.scripts/gptparser.py`
- `file:.scripts/idxtool.py`
- `file:.scripts/oneoff.py`
- `file:.vscode/launch.json`
- `file:.vscode/settings.json`
- `file:.github/workflows/build-toc.yaml`
- `file:.github/workflows/update-token-count.yml`
- `file:prompts/gpts/00GrDoGJY_Personality_Quiz_Creator.md`
- `file:prompts/gpts/02zmxuXd5_Node.js GPT - Project Builder.md`
- `file:prompts/gpts/03XS9XEyN_Nyxia_-_A_Spiritual_Cat.md`

## Potential Mislabeling Issues

10 texts flagged as potentially mislabeled.
These are texts labeled `is_injection=False` but contain suspicious keywords.

## Recommendations

1. **Fix text_length**: Currently 97% null - should be computed for all rows
2. **Remove suspicious sources**: Filter out internal paths (8110 sources)
3. **Manual label review**: Use `data/validation_samples.csv` to review 40 samples
4. **Drop unused columns**: `original_*` columns are 100% null
5. **Deduplicate**: Remove 0 duplicate text entries

## Next Steps

1. Run `python scripts/clean_data.py` to apply fixes
2. Review `data/validation_samples.csv` and update labels
3. Re-run validation after cleaning to verify fixes
