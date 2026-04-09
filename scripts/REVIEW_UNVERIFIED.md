# Unverified Submissions Review Script

## Overview
The `review_unverified.py` script provides an interactive interface for reviewing user submissions from the `unverified_user_submissions.parquet` file and moving verified entries to the main `prompts.parquet` database.

## Features

- **Interactive Review**: Presents each unverified submission with full context
- **Context Display**: Shows original model predictions, user labels, confidence scores, and metadata
- **Progress Tracking**: Saves reviewed IDs to resume interrupted sessions
- **Duplicate Prevention**: Checks if prompts already exist in the main database
- **Metadata Preservation**: Adds review metadata to migrated prompts
- **Statistics**: Tracks review progress and agreement rates

## Usage

### Basic Usage
```bash
cd /home/user/code/prompt-scan
uv run python scripts/review_unverified.py
```

### Command Line Options
```bash
uv run python scripts/review_unverified.py --help

Options:
  --unverified PATH    Path to unverified submissions parquet file
                       (default: website/api/data/unverified_user_submissions.parquet)
  --prompts PATH       Path to main prompts parquet file
                       (default: data/prompts.parquet)
  --progress PATH      Path to progress tracking file
                       (default: .reviewed_ids.json)
  --resume             Resume from previous progress
```

### Example Session
```
============================================================
🔍 UNVERIFIED SUBMISSIONS REVIEW
============================================================
Found 8 unreviewed entries
============================================================

📋 Entry 1/8 (ID: 1)

============================================================
📝 PROMPT:
  [Prompt text displayed here with wrapping...]

🔍 ORIGINAL CONTEXT:
  • Model predicted: SAFE (98.8% confidence)
  • User labeled: INJECTION
  • Status: ❌ DISAGREEMENT
  • Individual model predictions:
    - CNN: SAFE (99.8%)
    - LSTM: SAFE (100.0%)
    - Transformer: SAFE (96.5%)
  • Submitted: 2026-04-08 16:17:12
  • Source: web_interface
  • Model type: ensemble
============================================================

❓ DECISION:
  [y] Yes - This IS an injection
  [n] No - This is SAFE
  [s] Skip - Review later
  [q] Quit - Save and exit

Your choice (y/n/s/q):
```

## Decision Options

- **y**: Mark as injection → Added to main database as `is_injection=True`
- **n**: Mark as safe → Added to main database as `is_injection=False`
- **s**: Skip → Keep in unverified for later review
- **q**: Quit → Save progress and exit

## Data Flow

1. **Load**: Script loads entries from `unverified_user_submissions.parquet`
2. **Filter**: Excludes already reviewed entries (tracked in `.reviewed_ids.json`)
3. **Display**: Shows each entry with original context
4. **Decision**: User decides injection/safe/skip
5. **Migration**: Verified prompts added to `prompts.parquet` with metadata
6. **Cleanup**: Reviewed entries removed from unverified file
7. **Tracking**: Progress saved to allow resume

## Metadata Added

When a prompt is migrated to the main database, the following metadata is added:

- `source`: "user_feedback_reviewed"
- `review_date`: ISO timestamp of review
- `original_prediction`: What the model predicted
- `original_user_label`: What the user labeled
- `original_confidence`: Model's confidence score
- `original_source`: Original submission source
- `original_timestamp`: When the feedback was submitted

## Progress Tracking

The script maintains a progress file (default: `.reviewed_ids.json`) that contains:
- `reviewed_ids`: List of IDs that have been reviewed
- `last_updated`: Timestamp of last update
- `stats`: Review statistics (counts, agreement rates)

To resume an interrupted session, simply run the script again - it will automatically skip reviewed entries.

## Integration with Existing System

- Uses existing `ParquetFeedbackStore` and `ParquetDataStore` classes
- Compatible with current data schemas
- Follows duplicate prevention logic from main database
- Maintains data integrity with proper error handling

## Testing

A test script is available to verify functionality:
```bash
uv run python scripts/test_review.py
```

## Notes

- The script handles keyboard interrupts (Ctrl+C) gracefully and saves progress
- Duplicate prompts are detected and skipped
- Long prompts are truncated for display but stored in full
- Review decisions are final - once migrated, prompts cannot be "un-reviewed" through the script
- For batch operations or automation, consider extending the script or using the underlying classes directly